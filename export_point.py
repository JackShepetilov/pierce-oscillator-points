"""
ОДНА ТОЧКА СЕРИИ В ЯВНОМ ВИДЕ: сырые данные и каждый шаг обработки -> CSV + графики.

Запуск (из папки, где лежит этот файл):
    python export_point.py rect 10 100                  # форма, амплитуда A (В), ширина W (нс)
    python export_point.py путь\\к\\A10V_W100ns_raw.npz   # или прямо файл точки
Ключи:
    --sweep <папка>   где лежит серия (по умолчанию SWEEP ниже)
    --out <папка>     куда писать (по умолчанию <папка этого файла>\\<форма>_A..V_W..ns)

Метод тот же, что в infiniium_phase_stats.py (и check_by_hand.py) основного проекта, но сохраняется
каждый промежуточный результат, по каждому событию отдельно. Описание файлов — 00_README.txt
в папке точки, инструкция — README.md. Нужны только numpy и matplotlib. Серию не изменяет.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CH2_FIX = 10.0        # CH2: щуп был 1:1, а в настройке канала 10:1 -> вольты CH2 в файлах ×10. Делим.
WIN = 0.3e-6          # окно бегущей подгонки, с
STEP = 0.1e-6         # шаг окна, с
REC_PHASE, REC_AMP = 2.0, 0.05   # «восстановилось»: |Δφ| < 2° и |A/A0 − 1| < 5 %

SWEEP = r'C:\Матвей\Наука\Pierce_Oscillator\data\sweep_20261110_144320'   # главная серия 10.11.2026

ap = argparse.ArgumentParser(description='Одна точка серии в явном виде (CSV + графики)')
ap.add_argument('point', nargs='+', help='форма A W  (например: rect 10 100)  или путь к *_raw.npz')
ap.add_argument('--sweep', default=SWEEP, help='папка серии')
ap.add_argument('--out', help='папка вывода')
args = ap.parse_args()
if len(args.point) == 1:
    src = args.point[0]
elif len(args.point) == 3:
    s_, a_, w_ = args.point
    src = os.path.join(args.sweep, s_, f'A{float(a_):g}V_W{float(w_):g}ns_raw.npz')
else:
    ap.error('нужно либо 3 значения (форма A W), либо один путь к *_raw.npz')
if not os.path.isfile(src):
    folder = os.path.dirname(src)
    have = sorted(f[:-8] for f in os.listdir(folder) if f.endswith('_raw.npz')) if os.path.isdir(folder) else []
    sys.exit(f'Нет файла: {src}\n' + (f'В {folder} есть: ' + ', '.join(have) if have else
                                       'Проверьте форму (rect, haversine, triangle, gauss, sinecycle) и --sweep'))
src = os.path.abspath(src)
d = np.load(src)
shape, A, W = str(d['shape']), float(d['amp']), float(d['width'])
name = f'{shape}_A{A:g}V_W{W * 1e9:.0f}ns'
out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
os.makedirs(out, exist_ok=True)
P = lambda f: os.path.join(out, f)

t = d['t'].astype(float)
pulse = d['pulse'].astype(float)                 # CH1, В (верные)
osc = d['osc'].astype(float) / CH2_FIX           # CH2, В (исправлено ÷10)
nev, npt = osc.shape
dt = t[1] - t[0]
ev_cols = ','.join(f'ev{k:03d}' for k in range(1, nev + 1))
print(f'{name}: {nev} событий × {npt} точек, шаг {dt * 1e9:.2f} нс  ->  {out}')

# ---------- шаг 1. сырые данные ----------
for fn, X, what in [('01_raw_CH1_pulse_V.csv', pulse, 'CH1'), ('02_raw_CH2_osc_V.csv', osc, 'CH2')]:
    print(f'  пишу {fn} ...', flush=True)
    np.savetxt(P(fn), np.column_stack([t * 1e6, X.T]), delimiter=',', fmt='%.6g',
               header='t_us,' + ev_cols, comments='')

# ---------- шаг 2. момент фронта помехи в каждом событии ----------
level = min(max(A / 2, 1.3), 3.0)                # уровень запуска, как в серии
i0 = npt // 2 - npt // 20
t_edge = np.full(nev, np.nan)
for k in range(nev):
    above = pulse[k] > level
    idx = np.flatnonzero(~above[i0:-1] & above[i0 + 1:])
    if idx.size:
        i = idx[0] + i0
        t_edge[k] = t[i] + (level - pulse[k, i]) / (pulse[k, i + 1] - pulse[k, i]) * dt

# ---------- шаг 3. частота по участку ДО помехи ----------
pre = (t[0] + 0.1e-6, -0.1e-6 - 2.5 * W)
late = (max(4e-6, 3 * W + 3e-6), t[-1] - 0.1e-6)
mpre = (t >= pre[0]) & (t <= pre[1])


def peak_freq(x, tt):
    x = (x - x.mean()) * np.hanning(x.size)
    n = 16 * x.size
    X = np.abs(np.fft.rfft(x, n)); f = np.fft.rfftfreq(n, tt[1] - tt[0])
    lo = np.searchsorted(f, 1e6); k = np.argmax(X[lo:]) + lo
    y0, y1, y2 = np.log(X[k - 1:k + 2])
    return f[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (f[1] - f[0])


f_each = np.array([peak_freq(osc[k, mpre], t[mpre]) for k in range(nev)])
f0 = np.median(f_each)
w = 2 * np.pi * f0
np.savetxt(P('03_frequency.csv'), np.column_stack([np.arange(1, nev + 1), f_each]), delimiter=',',
           fmt=['%d', '%.3f'], header='event,f_Hz', comments='')


# ---------- шаг 4. «фантом»: синусоида по участку ДО помехи ----------
def fit(lo, hi):
    """osc[:, lo<=t<=hi] ≈ a·sin(wt) + b·cos(wt) + c  для всех событий сразу.
    a·sin+b·cos = |z|·sin(wt + arg z), z = a + i·b."""
    m = (t >= lo) & (t <= hi)
    M = np.column_stack([np.sin(w * t[m]), np.cos(w * t[m]), np.ones(m.sum())])
    C = np.linalg.lstsq(M, osc[:, m].T, rcond=None)[0]
    resid = np.std(osc[:, m] - (M @ C).T, axis=1)
    return C[0], C[1], C[2], resid


a0, b0, c0, r0 = fit(*pre)
z0 = a0 + 1j * b0
A0 = np.abs(z0)
phi0 = np.degrees(np.angle(z0))                               # фаза фантома при t = 0
phase_hit = (phi0 + 360 * f0 * t_edge) % 360                  # фаза генерации в момент фронта помехи
np.savetxt(P('04_phantom.csv'),
           np.column_stack([np.arange(1, nev + 1), t_edge * 1e9, a0, b0, c0, A0, phi0 % 360, phase_hit, r0]),
           delimiter=',', fmt=['%d'] + ['%.6g'] * 8,
           header='event,t_edge_ns,a_V,b_V,c_V,A0_V,phi0_deg,phase_hit_deg,resid_V', comments='')

# ---------- шаг 5. бегущее окно: фаза и амплитуда относительно фантома ----------
cs = np.arange(-2e-6, t[-1] - WIN / 2, STEP)
R = np.empty((nev, cs.size), complex)
for j, c in enumerate(cs):
    a, b, _, _ = fit(c - WIN / 2, c + WIN / 2)
    R[:, j] = (a + 1j * b) / z0
dphi_ev = np.degrees(np.angle(R))                             # (−180…180]
amp_ev = np.abs(R)
hdr = 'center_us,' + ev_cols
np.savetxt(P('05_track_dphi_deg_each.csv'), np.column_stack([cs * 1e6, dphi_ev.T]), delimiter=',',
           fmt='%.5g', header=hdr, comments='')
np.savetxt(P('06_track_amp_rel_each.csv'), np.column_stack([cs * 1e6, amp_ev.T]), delimiter=',',
           fmt='%.5g', header=hdr, comments='')

# ---------- шаг 6. среднее по событиям ----------
Rm = R.mean(0)                                                # комплексное среднее
dphi = np.degrees(np.angle(Rm))
dphi_sd = np.std(np.degrees(np.angle(R / Rm)), axis=0)
amp = amp_ev.mean(0)
amp_sd = amp_ev.std(0)
mean_pulse = pulse.mean(0) - np.median(pulse.mean(0)[t < -3e-6])
pul_at_c = np.interp(cs, t, mean_pulse)
np.savetxt(P('07_track_mean.csv'),
           np.column_stack([cs * 1e6, dphi, dphi_sd, amp, amp_sd, pul_at_c]), delimiter=',', fmt='%.6g',
           header='center_us,dphi_mean_deg,dphi_sd_deg,amp_mean_rel,amp_sd_rel,mean_pulse_V', comments='')

# ---------- шаг 7. постоянный сдвиг (окно «поздно») ----------
al, bl, _, _ = fit(*late)
zl = al + 1j * bl
shift = np.degrees(np.angle(zl / z0))
amp_late = np.abs(zl) / A0
zone = (cs > -0.5e-6) & (cs < late[0])
jm = np.flatnonzero(zone & (amp > 0.5))[np.argmax(np.abs(dphi[zone & (amp > 0.5)]))]   # пик среднего отклонения
ja = np.flatnonzero(zone)[np.argmin(amp[zone])]                                       # провал амплитуды
np.savetxt(P('08_per_event_result.csv'),
           np.column_stack([np.arange(1, nev + 1), phase_hit, dphi_ev[:, jm], amp_ev[:, ja], amp_late, shift]),
           delimiter=',', fmt=['%d'] + ['%.6g'] * 5,
           header=f'event,phase_hit_deg,dphi_at_{cs[jm] * 1e6:.1f}us_deg,amp_at_{cs[ja] * 1e6:.1f}us_rel,'
                  f'amp_late_rel,shift_late_deg', comments='')

# ---------- шаг 8. итоговые числа ----------
bad = np.flatnonzero(zone & ((np.abs(dphi) > REC_PHASE) | (np.abs(amp - 1) > REC_AMP)))
t_rec = cs[bad[-1]] + WIN / 2 if bad.size else 0.0


def harm1(ph, y):
    """y ≈ k0 + k1·cos φ + s1·sin φ -> (k0, размах 1-й гармоники, φ её максимума)."""
    r = np.radians(ph); M = np.column_stack([np.ones_like(r), np.cos(r), np.sin(r)])
    (k0, k1, s1), *_ = np.linalg.lstsq(M, y, rcond=None)
    return k0, np.hypot(k1, s1), np.degrees(np.arctan2(s1, k1)) % 360


_, prc_a, prc_p = harm1(phase_hit, shift)
pk, dip = dphi_ev[:, jm], amp_ev[:, ja]      # по каждому событию: Δφ в пике и A/A0 в провале
R_hit = abs(np.mean(np.exp(1j * np.radians(phase_hit))))
shift_mean = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(shift)))))
res = [
    ('файл', src), ('форма', shape), ('A заказ, В', A), ('W заказ, нс', W * 1e9),
    ('событий', nev), ('событий с найденным фронтом', int(np.isfinite(t_edge).sum())),
    ('уровень поиска фронта CH1, В', level),
    ('окно ДО, мкс', f'{pre[0] * 1e6:.2f} … {pre[1] * 1e6:.2f}'),
    ('окно ПОЗДНО, мкс', f'{late[0] * 1e6:.2f} … {late[1] * 1e6:.2f}'),
    ('частота f0 (медиана), Гц', f'{f0:.1f}'), ('разброс частоты по событиям, Гц', f'{f_each.std():.1f}'),
    ('амплитуда генерации до помехи (медиана), В', f'{np.median(A0):.4f}'),
    ('шум подгонки фантома (медиана), В', f'{np.median(r0):.4f}'),
    ('равномерность фазы попадания R (0 = равномерно)', f'{R_hit:.3f}'),
    ('макс. среднее отклонение фазы (где амплитуда > 0,5), °', f'{dphi[jm]:+.2f}'),
    ('  ... при t, мкс', f'{cs[jm] * 1e6:.1f}'), ('  ... разброс между событиями, °', f'{dphi_sd[jm]:.2f}'),
    ('  ... по событиям: от … до, ° (при фазе попадания, °)',
     f'{pk.min():+.1f} ({phase_hit[pk.argmin()]:.0f}) … {pk.max():+.1f} ({phase_hit[pk.argmax()]:.0f})'),
    ('мин. средняя амплитуда (доля исходной)', f'{amp[ja]:.3f}'), ('  ... при t, мкс', f'{cs[ja] * 1e6:.1f}'),
    ('  ... по событиям: от … до (при фазе попадания, °)',
     f'{dip.min():.3f} ({phase_hit[dip.argmin()]:.0f}) … {dip.max():.3f} ({phase_hit[dip.argmax()]:.0f})'),
    ('время восстановления (|Δφ|<2° и |A−1|<5 %), мкс', f'{t_rec * 1e6:.2f}'),
    ('постоянный сдвиг фазы (среднее), °', f'{shift_mean:+.4f}'),
    ('  ... ошибка среднего, °', f'{shift.std() / np.sqrt(nev):.4f}'),
    ('  ... разброс между событиями, °', f'{shift.std():.4f}'),
    ('  ... зависимость от фазы попадания (PRC): ±°, макс. при °', f'{prc_a:.3f}, {prc_p:.0f}'),
    ('амплитуда поздно / до (медиана)', f'{np.median(amp_late):.4f}'),
]
with open(P('09_result.txt'), 'w', encoding='utf-8') as fo:
    for k, v in res:
        fo.write(f'{k}: {v}\n')
for k, v in res[9:]:
    print(f'  {k}: {v}')

# ---------- графики ----------
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3})
T = t * 1e6
# 1. одно сырое событие
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5))
ax[0].plot(T, pulse[0], 'r', lw=.7); ax[1].plot(T, osc[0], 'g', lw=.7)
ax[0].set_ylabel('CH1 помеха, В'); ax[1].set_ylabel('CH2 генерация, В (÷10)'); ax[1].set_xlabel('t, мкс')
ax[1].set_xlim(-1, 3); ax[0].set_title(f'{name}: событие 1 из {nev} (столбец ev001), сырые данные')
fig.tight_layout(); fig.savefig(P('fig1_raw.png'), dpi=150); plt.close(fig)
# 2. одно событие + фантом
k = 0
fig, ax = plt.subplots(figsize=(10, 3.8))
z = (t > -0.6e-6) & (t < 2.5e-6)
ax.plot(T[z], osc[k, z], lw=.8, label='CH2, событие 1')
ax.plot(T[z], a0[k] * np.sin(w * t[z]) + b0[k] * np.cos(w * t[z]) + c0[k], 'k--', lw=.8,
        label='фантом (подгонка по участку до помехи)')
ax.axvline(pre[1] * 1e6, color='g', ls=':', label='конец окна ДО')
ax.set_xlabel('t, мкс'); ax.set_ylabel('В'); ax.legend(fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(P('fig2_phantom.png'), dpi=150); plt.close(fig)
# 3. ход фазы/амплитуды: каждое событие + среднее
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7.5), gridspec_kw=dict(height_ratios=[1, 2, 2]))
ax[0].plot(T, mean_pulse, 'r', lw=.8); ax[0].set_ylabel('помеха, В')
good = amp_ev > 0.5
ax[1].plot(cs * 1e6, np.where(good, dphi_ev, np.nan).T, color='0.6', lw=.3)
ax[1].plot(cs * 1e6, np.where(amp > 0.5, dphi, np.nan), 'k', lw=1.5, label='среднее')
ax[1].set_ylabel('Δφ к фантому, °'); ax[1].legend(fontsize=7)
ax[2].plot(cs * 1e6, amp_ev.T, color='0.6', lw=.3); ax[2].plot(cs * 1e6, amp, 'b', lw=1.5)
ax[2].set_ylabel('A / A0'); ax[2].set_xlabel('центр окна, мкс от запуска')
ax[2].axvline(t_rec * 1e6, color='g', ls=':'); ax[1].axvline(t_rec * 1e6, color='g', ls=':')
ax[0].set_title(f'{name}: серые — каждое из {nev} событий, жирная — среднее; зелёная — восстановление')
ax[2].set_xlim(-2, 8)
fig.tight_layout(); fig.savefig(P('fig3_tracks.png'), dpi=150); plt.close(fig)
# 4. карта: события, отсортированные по фазе попадания
o = np.argsort(phase_hit)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
ext = [cs[0] * 1e6, cs[-1] * 1e6, 0, nev]
im = ax[0].imshow(np.where(good, dphi_ev, np.nan)[o], aspect='auto', origin='lower', extent=ext,
                  cmap='RdBu_r', vmin=-60, vmax=60)
fig.colorbar(im, ax=ax[0], label='Δφ, °')
im = ax[1].imshow(amp_ev[o], aspect='auto', origin='lower', extent=ext, cmap='viridis', vmin=0, vmax=1.1)
fig.colorbar(im, ax=ax[1], label='A / A0')
for a_ in ax:
    a_.set_xlim(-1, 4); a_.set_xlabel('мкс')
ax[0].set_ylabel('события по возрастанию фазы попадания (0…360°)')
fig.tight_layout(); fig.savefig(P('fig4_map_by_hit_phase.png'), dpi=150); plt.close(fig)
# 5. зависимость от фазы попадания
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a_, y, ttl in [(ax[0], pk, f'Δφ при {cs[jm] * 1e6:.1f} мкс, °'),
                   (ax[1], dip, f'A/A0 при {cs[ja] * 1e6:.1f} мкс'),
                   (ax[2], shift, 'постоянный сдвиг (поздно), °')]:
    a_.plot(phase_hit, y, '.', ms=3)       # только точки, без аппроксимации
    a_.set_xlabel('фаза попадания, °'); a_.set_title(ttl, fontsize=9); a_.set_xlim(0, 360)
fig.tight_layout(); fig.savefig(P('fig5_vs_hit_phase.png'), dpi=150); plt.close(fig)

# ---------- README ----------
readme = f"""Точка {name} в явном виде.  Источник: {src}
Сделано scripts/export_point.py. Все CSV: разделитель — запятая, десятичная точка, первая строка — заголовки.
MATLAB: T = readtable('02_raw_CH2_osc_V.csv');  или  M = readmatrix('02_raw_CH2_osc_V.csv');  (1-й столбец — время)
Время 0 = момент запуска осциллографа по фронту помехи (CH1). Событие = один импульс помехи (всего {nev}).

ВАЖНО: CH2 в этих файлах уже ПОДЕЛЁН на 10 (щуп был 1:1, а в настройке канала 10:1).
В исходных *_raw.npz и в summary.csv серии вольты CH2 ×10. Фазы и относительные амплитуды от этого не зависят.
CH1 при A ≥ 7 В обрезан сверху на ~6,4 В (край экрана) — на поиск фронта не влияет.

ШАГ 1. Сырые данные
  01_raw_CH1_pulse_V.csv   t_us, ev001…ev{nev:03d} — помеха (CH1), В. {npt} строк, шаг {dt * 1e9:.2f} нс.
  02_raw_CH2_osc_V.csv     t_us, ev001…ev{nev:03d} — генерация (CH2, сток), В (уже ÷10).

ШАГ 2. Фронт помехи: первое пересечение CH1 уровня {level:g} В снизу вверх около t = 0,
        с линейной интерполяцией между отсчётами  ->  столбец t_edge_ns в 04_phantom.csv.

ШАГ 3. Частота генерации
  03_frequency.csv         event, f_Hz — частота каждого события: БПФ участка ДО помехи
                           (окно Ханна, дополнение нулями ×16, уточнение пика параболой по логарифму).
                           Дальше везде одна частота f0 = медиана = {f0:.1f} Гц.
  Зачем так точно: ошибка δf даёт ложный наклон фазы 360°·δf·Δt (10 Гц за 10 мкс -> 0,04°).

ШАГ 4. «Фантом» — продолжение невозмущённой генерации
  04_phantom.csv           по участку ДО ({pre[0] * 1e6:.2f} … {pre[1] * 1e6:.2f} мкс) МНК-подгонка
                           CH2 ≈ a·sin(2πf0·t) + b·cos(2πf0·t) + c   (линейная, без начальных приближений).
                           A0 = √(a²+b²) — амплитуда; phi0 = atan2(b, a) — фаза при t = 0;
                           phase_hit = phi0 + 360·f0·t_edge — фаза генерации в момент фронта помехи;
                           resid_V — СКО остатка подгонки (шум + гармоники).

ШАГ 5. Бегущее окно {WIN * 1e6:.1f} мкс с шагом {STEP * 1e6:.1f} мкс — та же подгонка (a, b) в каждом окне.
        Сравнение с фантомом:  r = (a + i·b) / (a0 + i·b0)
  05_track_dphi_deg_each.csv   center_us, ev001… — Δφ = arg r, ° (сдвиг фазы этого события к фантому)
  06_track_amp_rel_each.csv    center_us, ev001… — |r| = A/A0 (амплитуда к исходной)
  Где A/A0 < 0,5, фаза определена плохо (генерация почти заглохла) — на графиках там пропуск.

ШАГ 6. Среднее по событиям
  07_track_mean.csv        dphi_mean_deg = arg(среднее r) — комплексное среднее;
                           dphi_sd_deg — разброс Δφ между событиями; amp_mean_rel, amp_sd_rel — то же для |r|;
                           mean_pulse_V — средняя помеха (наводка генерации со случайной фазой гасится).

ШАГ 7. По каждому событию
  08_per_event_result.csv  phase_hit_deg; Δφ в момент пика среднего отклонения; A/A0 в момент провала;
                           amp_late_rel и shift_late_deg — амплитуда и постоянный сдвиг фазы по окну ПОЗДНО
                           ({late[0] * 1e6:.2f} … {late[1] * 1e6:.2f} мкс), сравнение с фантомом.

ШАГ 8. Итог
  09_result.txt            все итоговые числа точки. Время восстановления — конец последнего окна, где
                           |Δφ среднее| > {REC_PHASE:g}° или |A/A0 − 1| > {REC_AMP * 100:g} %.
                           Для пика Δφ и провала амплитуды дан диапазон по событиям (min … max) и фаза попадания,
                           где он достигается. Для постоянного сдвига (PRC) — подгонка y = k0 + k·cos(φ_hit − φ_max),
                           даётся ±k и φ_max (только числом, на графике её нет). Осторожно: сама подгонка синусоиды
                           по окнам даёт ложную «зависимость» порядка ±0,01…0,14° (проверено на точке rect 10 В/100 нс
                           сравнением двух окон без помехи), поэтому ±k такого размера — не эффект.

Графики
  fig1_raw.png             одно сырое событие (ev001): помеха и генерация
  fig2_phantom.png         одно событие и его фантом
  fig3_tracks.png          ход Δφ и A/A0: каждое событие (серые) и среднее (жирные)
  fig4_map_by_hit_phase.png  то же картой: строки — события по возрастанию фазы попадания
  fig5_vs_hit_phase.png    Δφ в пике, провал амплитуды и постоянный сдвиг против фазы попадания
                           (каждая точка — одно событие, без аппроксимации)
"""
with open(P('00_README.txt'), 'w', encoding='utf-8') as fo:
    fo.write(readme)
print('Готово.')
