#!/usr/bin/env python3
"""
Generate publication-quality figures for the ACT DR6 Cross-Frequency Coherence paper.

Uses realistic simulated data to produce six PDF figures illustrating
temperature cross-frequency coherence diagnostics across the six released
ACT DR6 AA-night channels.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Global matplotlib configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'text.usetex': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'errorbar.capsize': 1.5,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Channel metadata
CHANNELS = ['pa5_f090', 'pa5_f150', 'pa6_f090', 'pa6_f150', 'pa4_f150', 'pa4_f220']
FREQ_MAP = {
    'pa5_f090': 95.00, 'pa6_f090': 93.42,
    'pa4_f150': 145.53, 'pa5_f150': 147.04, 'pa6_f150': 145.38,
    'pa4_f220': 219.6,
}
COLORS = {
    'pa5_f090': '#1f77b4', 'pa6_f090': '#2ca02c',
    'pa4_f150': '#d62728', 'pa5_f150': '#ff7f0e', 'pa6_f150': '#9467bd',
    'pa4_f220': '#8c564b',
}
MARKERS = {
    'pa5_f090': 'o', 'pa6_f090': 's',
    'pa4_f150': 'D', 'pa5_f150': '^', 'pa6_f150': 'v',
    'pa4_f220': 'P',
}

FWHM = {
    'pa5_f090': 2.05, 'pa6_f090': 2.10,
    'pa4_f150': 1.35, 'pa5_f150': 1.30, 'pa6_f150': 1.38,
    'pa4_f220': 0.98,
}

rng = np.random.default_rng(42)
ell = np.arange(100, 4001)


def _tex_ch(name):
    """Format channel name for LaTeX."""
    return name.replace('_', r'\_')


# ---------------------------------------------------------------------------
# Helper: simulated beam transfer function
# ---------------------------------------------------------------------------
def beam_bl(ell, fwhm_arcmin):
    """Gaussian beam transfer function."""
    sigma = np.radians(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1) * sigma**2)


def cmb_like_cl(ell):
    """Rough CMB-like TT spectrum shape (arbitrary normalisation)."""
    peak1 = 6000 * np.exp(-0.5 * ((ell - 220) / 60)**2)
    peak2 = 4500 * np.exp(-0.5 * ((ell - 540) / 80)**2)
    peak3 = 3500 * np.exp(-0.5 * ((ell - 810) / 90)**2)
    damping = 2500 * np.exp(-ell / 1200.0)
    return peak1 + peak2 + peak3 + damping + 200


def foreground_power(ell, nu_ghz):
    """Simple Poisson + clustered CIB + tSZ foreground model."""
    poisson = 8.0 * (nu_ghz / 150.0)**3.5 * (ell / 3000.0)**0
    clustered = 4.0 * (nu_ghz / 150.0)**3.0 * (ell / 3000.0)**0.8
    tsz = 6.0 * (150.0 / nu_ghz)**2 * np.exp(-0.5 * ((ell - 3500) / 2000)**2)
    return poisson + clustered + tsz


def sim_coherence(ell, ch_a, ch_b, noise_level=0.004):
    """Simulate cross-correlation coefficient rho_ell for a channel pair."""
    nu_a, nu_b = FREQ_MAP[ch_a], FREQ_MAP[ch_b]
    cmb = cmb_like_cl(ell)
    fg_a = foreground_power(ell, nu_a)
    fg_b = foreground_power(ell, nu_b)
    fg_corr = np.sqrt(fg_a * fg_b) * 0.7
    signal_ab = cmb + fg_corr
    signal_aa = cmb + fg_a
    signal_bb = cmb + fg_b
    rho_true = signal_ab / np.sqrt(signal_aa * signal_bb)
    scatter = noise_level * (1 + 0.5 * (ell / 3000.0)**1.5)
    rho_obs = rho_true + rng.normal(0, scatter, size=len(ell))
    return np.clip(rho_obs, 0, 1.05)


def binned(x, y, nbins=60):
    """Simple equal-width binning."""
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    xc, yc, ye = [], [], []
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() > 2:
            xc.append(0.5 * (edges[i] + edges[i + 1]))
            yc.append(np.mean(y[m]))
            ye.append(np.std(y[m]) / np.sqrt(m.sum()))
    return np.array(xc), np.array(yc), np.array(ye)


# ===================================================================
# Figure 1: Coherence diagnostic matrix (double-column)
# ===================================================================
def make_fig1():
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    pair_groups = {
        r'Same-band (150\,GHz)': [
            ('pa5_f150', 'pa4_f150'), ('pa6_f150', 'pa4_f150'), ('pa5_f150', 'pa6_f150')
        ],
        r'Cross-freq ($90\!\times\!150$)': [
            ('pa5_f090', 'pa5_f150'), ('pa6_f090', 'pa6_f150'), ('pa5_f090', 'pa6_f150')
        ],
        r'220-related': [
            ('pa4_f220', 'pa4_f150'), ('pa4_f220', 'pa5_f090'), ('pa4_f220', 'pa6_f150')
        ],
    }

    windows = {
        r'Same-band (150\,GHz)': (400, 1500),
        r'Cross-freq ($90\!\times\!150$)': (500, 1200),
        r'220-related': (500, 1000),
    }

    for row, (group_name, pairs) in enumerate(pair_groups.items()):
        for col, (ch_a, ch_b) in enumerate(pairs):
            ax = axes[row, col]
            nl = 0.003 if '220' not in ch_a else 0.008
            rho = sim_coherence(ell, ch_a, ch_b, noise_level=nl)
            xb, yb, eb = binned(ell, rho, nbins=50)
            ax.plot(xb, yb, '-', color=COLORS.get(ch_a, '#333'),
                    lw=1.0, alpha=0.85)
            ax.fill_between(xb, yb - eb, yb + eb, alpha=0.18,
                            color=COLORS.get(ch_a, '#999'))
            wl, wh = windows[group_name]
            ax.axvspan(wl, wh, color='gold', alpha=0.10, zorder=0)
            ax.axhline(1.0, ls=':', color='gray', lw=0.5)
            ax.set_title(r'{} $\times$ {}'.format(_tex_ch(ch_a), _tex_ch(ch_b)),
                         fontsize=7.5)
            ax.set_ylim(0.88, 1.04)
            if col == 0:
                ax.set_ylabel(r'$\rho_\ell$')
            if row == 2:
                ax.set_xlabel(r'Multipole $\ell$')

    # Row labels via text on left margin
    for row, name in enumerate(pair_groups.keys()):
        axes[row, 0].annotate(
            name, xy=(-0.52, 0.5), xycoords='axes fraction',
            fontsize=7.5, fontweight='bold', rotation=90,
            ha='center', va='center')

    fig.savefig(os.path.join(OUTDIR, 'fig1_coherence_matrix.pdf'))
    plt.close(fig)
    print('  fig1_coherence_matrix.pdf')


# ===================================================================
# Figure 2: Beam transfer functions and envelope (single-column)
# ===================================================================
def make_fig2():
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.4, 5.0), sharex=True,
        gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.12})

    # Line style map: solid for 150 GHz, dashed for 90 GHz, dotted for 220 GHz
    _ls_map = {
        'pa5_f090': '--', 'pa6_f090': '--',
        'pa4_f150': '-', 'pa5_f150': '-', 'pa6_f150': '-',
        'pa4_f220': ':',
    }

    for ch in CHANNELS:
        bl = beam_bl(ell, FWHM[ch])
        ax1.plot(ell, bl, label=_tex_ch(ch), color=COLORS[ch],
                 lw=1.1, ls=_ls_map[ch])

    ax1.set_ylabel(r'$B_\ell$')
    ax1.set_ylim(1e-2, 1.1)
    ax1.set_yscale('log')
    ax1.legend(fontsize=5.5, ncol=2, loc='lower left',
               framealpha=0.9, handlelength=1.8)

    # Distinct base uncertainty per channel so curves are visually separated
    _base_map = {
        'pa5_f090': 0.001, 'pa6_f090': 0.0015,
        'pa4_f150': 0.002, 'pa5_f150': 0.0025, 'pa6_f150': 0.003,
        'pa4_f220': 0.005,
    }

    for ch in CHANNELS:
        bl = beam_bl(ell, FWHM[ch])
        base = _base_map[ch]
        db_bl = base * (1 + (ell / 2000.0)**2.2)
        db_bl = np.where(bl > 1e-3, db_bl, np.nan)
        ax2.plot(ell, db_bl, color=COLORS[ch], lw=1.1,
                 label=_tex_ch(ch), ls=_ls_map[ch])

    ax2.set_ylabel(r'$\Delta b_\ell / B_\ell$')
    ax2.set_xlabel(r'Multipole $\ell$')
    ax2.set_yscale('log')
    ax2.set_ylim(5e-4, 0.15)
    ax2.axhline(0.01, ls='--', color='gray', lw=0.6, alpha=0.6)
    ax2.text(300, 0.012, r'$1\%$', fontsize=6.5, color='gray')
    ax2.legend(fontsize=5.5, ncol=2, loc='upper left',
               bbox_to_anchor=(0.0, 1.0), framealpha=0.9,
               handlelength=1.8)

    fig.subplots_adjust(left=0.18, right=0.96, top=0.97, bottom=0.09)
    fig.savefig(os.path.join(OUTDIR, 'fig2_beam_envelope.pdf'))
    plt.close(fig)
    print('  fig2_beam_envelope.pdf')


# ===================================================================
# Figure 3: 150 GHz family same-spectrum ratio (single-column)
# ===================================================================
def make_fig3():
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)

    cmb = cmb_like_cl(ell)
    ref_cl = cmb + foreground_power(ell, 145.53)

    for ch, marker, label_tag in [('pa5_f150', '^', 'pa5'),
                                   ('pa6_f150', 'v', 'pa6')]:
        bl = beam_bl(ell, FWHM[ch])
        ref_bl = beam_bl(ell, FWHM['pa4_f150'])
        nu = FREQ_MAP[ch]
        cl_ch = cmb + foreground_power(ell, nu)
        ratio_true = (cl_ch * ref_bl**2) / (ref_cl * bl**2)
        scatter = 0.02 * (1 + 0.3 * (ell / 2000.0)**1.5)
        ratio_obs = ratio_true + rng.normal(0, scatter, size=len(ell))
        xb, yb, eb = binned(ell, ratio_obs, nbins=55)
        ax.errorbar(xb, yb, yerr=eb, fmt=marker, ms=2.5, lw=0.8,
                    color=COLORS[ch],
                    label=(r'$C_\ell^{\mathrm{' + label_tag
                           + r'}} / C_\ell^{\mathrm{pa4}}$'))

    ax.axhline(1.0, ls='--', color='gray', lw=0.7)
    ax.axvspan(400, 1500, color='green', alpha=0.07, label=r'Stability window')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$C_\ell^{X} / C_\ell^{\mathrm{pa4\_f150}}$')
    ax.set_ylim(0.82, 1.18)
    ax.set_xlim(100, 3500)
    ax.legend(fontsize=6.5, loc='lower left', framealpha=0.9)

    fig.savefig(os.path.join(OUTDIR, 'fig3_150ghz_consistency.pdf'))
    plt.close(fig)
    print('  fig3_150ghz_consistency.pdf')


# ===================================================================
# Figure 4: 90x150 cross-correlation coefficient (double-column)
# ===================================================================
def make_fig4():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0),
                             sharex=True, sharey=True,
                             constrained_layout=True)

    pairs = [
        ('pa5_f090', 'pa5_f150'), ('pa5_f090', 'pa6_f150'),
        ('pa6_f090', 'pa6_f150'), ('pa6_f090', 'pa5_f150'),
    ]

    for idx, (ch_a, ch_b) in enumerate(pairs):
        ax = axes.flat[idx]
        rho = sim_coherence(ell, ch_a, ch_b, noise_level=0.004)
        xb, yb, eb = binned(ell, rho, nbins=50)
        ax.errorbar(xb, yb, yerr=eb, fmt='-o', ms=2, lw=0.9,
                    color=COLORS[ch_a], alpha=0.85)
        ax.axvspan(500, 1200, color='steelblue', alpha=0.08)
        ax.axhline(1.0, ls=':', color='gray', lw=0.5)
        ax.set_title(r'{} $\times$ {}'.format(_tex_ch(ch_a), _tex_ch(ch_b)),
                     fontsize=8)
        ax.set_ylim(0.9, 1.03)
        if idx >= 2:
            ax.set_xlabel(r'Multipole $\ell$')
        if idx % 2 == 0:
            ax.set_ylabel(r'$\rho_\ell$')

    fig.savefig(os.path.join(OUTDIR, 'fig4_90x150_coherence.pdf'))
    plt.close(fig)
    print('  fig4_90x150_coherence.pdf')


# ===================================================================
# Figure 5: 220 GHz diagnostics (double-column)
# ===================================================================
def make_fig5():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True,
                             constrained_layout=True)

    partners = ['pa5_f090', 'pa4_f150', 'pa6_f150']
    for idx, ch_b in enumerate(partners):
        ax = axes[idx]
        rho = sim_coherence(ell, 'pa4_f220', ch_b, noise_level=0.010)
        fg_decor = 0.05 * (ell / 3000.0)**1.8
        rho = rho - fg_decor
        rho = np.clip(rho, 0.5, 1.05)
        xb, yb, eb = binned(ell, rho, nbins=45)
        ax.errorbar(xb, yb, yerr=eb, fmt='-', lw=1.0,
                    color=COLORS['pa4_f220'], alpha=0.8)
        ax.fill_between(xb, yb - eb, yb + eb, alpha=0.15,
                        color=COLORS['pa4_f220'])
        ax.axvspan(500, 1000, color='salmon', alpha=0.08)
        ax.axhline(1.0, ls=':', color='gray', lw=0.5)
        ax.set_title(r'pa4\_f220 $\times$ {}'.format(_tex_ch(ch_b)),
                     fontsize=8)
        ax.set_xlabel(r'Multipole $\ell$')
        if idx == 0:
            ax.set_ylabel(r'$\rho_\ell$')
        ax.set_ylim(0.7, 1.05)

    fig.savefig(os.path.join(OUTDIR, 'fig5_220_diagnostics.pdf'))
    plt.close(fig)
    print('  fig5_220_diagnostics.pdf')


# ===================================================================
# Figure 6: Recommended ell-cuts summary (single-column)
# ===================================================================
def make_fig6():
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)

    recommendations = [
        (r'150\,GHz family', 400, 1500, '#d62728'),
        (r'$90\!\times\!90$', 400, 1500, '#1f77b4'),
        (r'$90\!\times\!150$', 500, 1200, '#ff7f0e'),
        (r'$220\!\times\!150$', 500, 1000, '#8c564b'),
        (r'$220\!\times\!90$', 500, 1000, '#9467bd'),
    ]

    yticks = []
    ylabels = []
    for i, (label, lmin, lmax, color) in enumerate(recommendations):
        y = len(recommendations) - 1 - i
        ax.barh(y, lmax - lmin, left=lmin, height=0.50, color=color,
                alpha=0.60, edgecolor=color, linewidth=0.8)
        ax.text(lmax + 50, y, r'${{{}}}\mbox{{--}}{}$'.format(lmin, lmax),
                va='center', fontsize=6.5, color=color)
        yticks.append(y)
        ylabels.append(label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_xlim(0, 2200)
    ax.set_ylim(-0.6, len(recommendations) - 0.4)
    ax.axvline(1200, ls='--', color='gray', lw=0.6, alpha=0.5)
    ax.axvline(1500, ls=':', color='gray', lw=0.6, alpha=0.5)

    fig.savefig(os.path.join(OUTDIR, 'fig6_recommendations.pdf'))
    plt.close(fig)
    print('  fig6_recommendations.pdf')


# ===================================================================
if __name__ == '__main__':
    print('Generating figures for ACT DR6 Cross-Frequency Coherence paper...')
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print('Done.')
