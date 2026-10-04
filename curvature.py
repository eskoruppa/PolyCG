
import numpy as np
import polycg as pcg
import os, json, hashlib
from typing import Sequence


def dnacurv_chunked(
    seq: str,
    span: int | Sequence[int],
    model: str = 'cgna+',
    sglspan: int = 2000,
    overlap: int = 500,
    ) -> np.ndarray | list:
    """Sliding window bend angles for a sequence too long to be treated in one piece.

    The sequence is cut into chunks of sglspan bp advancing by sglspan-overlap bp,
    every chunk is evaluated on its own and the pieces are stitched back together.

    span may be a list or array of spans. All of them are evaluated from the same set
    of chunks, so the expensive configuration generation is shared. Since a chunk
    yields chunklen-span values, larger spans yield fewer of them and neighbouring
    chunks then share correspondingly less. Every span is therefore stitched
    separately, with its own trimming.

    Returns the raw cos(theta) values, one per reference base pair, i.e. len(seq)-span
    values, as a single array for a scalar span and as a list of arrays, one per
    requested span, otherwise.
    """
    single = np.ndim(span) == 0
    spans = [int(span)] if single else [int(s) for s in span]
    if len(spans) == 0:
        raise ValueError('no span provided')
    if min(spans) < 1:
        raise ValueError(f'span needs to be at least 1, encountered {min(spans)}')
    if max(spans) >= sglspan:
        raise ValueError(f'sglspan ({sglspan}) needs to exceed the largest span ({max(spans)})')
    if max(spans) > overlap:
        raise ValueError(f'overlap ({overlap}) needs to be at least the largest span ({max(spans)}), '
                         f'neighbouring chunks would otherwise not overlap in reference base pairs')

    stride = sglspan - overlap
    # bounded by the largest span, so that every chunk contributes to every span
    starts = list(range(0,len(seq)-max(spans),stride))

    # chunks[k][si]: values of span spans[si] within chunk k
    chunks = []
    for k,start in enumerate(starts):
        print(f'Processing sequence {k+1}/{len(starts)}')
        crvdct = pcg.dnacurv(seq[start:start+sglspan],spans,model=model,angles=False)
        chunks.append(crvdct['curvature'])

    # dnacurv returns chunklen-span values, value j referring to base pair start+j. The
    # redundancy between neighbouring chunks is hence overlap-span and shrinks as the
    # span grows. It is read off the actual reference base pair ranges rather than
    # assumed, so that every span is trimmed by its own amount. Half of the redundancy
    # is dropped on either side of a junction, so that both chunks shed their
    # terminus-affected edge rather than one of them being kept in full.
    crvs = []
    for si in range(len(spans)):
        segs = []
        for k in range(len(starts)):
            crv_k = chunks[k][si]
            lcut = rcut = 0
            if k > 0:
                ov = starts[k-1] + len(chunks[k-1][si]) - starts[k]
                lcut = ov - ov//2
            if k < len(starts)-1:
                ov = starts[k] + len(crv_k) - starts[k+1]
                rcut = ov//2
            segs.append(crv_k[lcut:len(crv_k)-rcut])
        crvs.append(np.concatenate(segs))

    return crvs[0] if single else crvs


def dnacurv_cached(
    fn: str,
    span: int | Sequence[int],
    model: str = 'cgna+',
    angles: bool = True,
    degrees: bool = True,
    sglspan: int = 2000,
    overlap: int = 500,
    cachesubdir: str | None = None,
    recompute: bool = False,
    ) -> np.ndarray | list:
    """dnacurv_chunked for the sequence in fn, cached on disk per sequence, span and model.

    The first call runs the full calculation and stores the result next to fn, in a
    subdirectory named after the sequence file with dots replaced by underscores and
    _cache appended, every later call reads it back instead. cachesubdir overrides
    that name and is taken as the cache directory as is if it is absolute.

    What is stored are the raw cos(theta) values, so angles and degrees are applied on
    the way out and toggling them costs nothing. A changed sequence, sglspan or overlap
    does invalidate the cache, and recompute=True forces a fresh run.

    span may be a list or array of spans. Every span is cached in its own file, so a
    later request for one of them is served from disk no matter which set it was
    generated with. Only the spans that are missing are computed, and those are
    computed in a single pass through the sequence.

    Returns a single array for a scalar span and a list of arrays, one per requested
    span, otherwise.
    """
    with open(fn,'r') as f:
        seq = f.read().strip()

    single = np.ndim(span) == 0
    spans = [int(span)] if single else [int(s) for s in span]

    base = os.path.splitext(os.path.basename(fn))[0].replace('.','_')
    seqhash = hashlib.md5(seq.encode()).hexdigest()
    if cachesubdir is None:
        cachesubdir = os.path.basename(fn).replace('.','_') + '_cache'
    # next to the sequence file, so that the cache is found independently of the
    # working directory. An absolute cachesubdir overrides this.
    cachedir = os.path.join(os.path.dirname(os.path.abspath(fn)),cachesubdir)

    def cachefile(s):
        return os.path.join(cachedir,f'{base}_{model.replace("+","plus")}_span{s}.npz')

    def metadata(s):
        return {
            'seqhash': seqhash,
            'nbp'    : len(seq),
            'span'   : s,
            'model'  : model,
            'sglspan': sglspan,
            'overlap': overlap,
            }

    crvs = {}
    for s in spans:
        cachefn = cachefile(s)
        if not os.path.isfile(cachefn) or recompute:
            continue
        meta = metadata(s)
        with np.load(cachefn) as dat:
            cachedmeta = json.loads(dat['meta'].item())
            if cachedmeta == meta:
                print(f'reading curvature from {cachefn}')
                crvs[s] = dat['curvature']
        if s not in crvs:
            changed = [key for key in meta if cachedmeta.get(key) != meta[key]]
            print(f'{cachefn} was generated with different settings ({", ".join(changed)}), recomputing')

    # dict.fromkeys to preserve the order while dropping repeated spans
    missing = [s for s in dict.fromkeys(spans) if s not in crvs]
    if len(missing) > 0:
        newcrvs = dnacurv_chunked(seq,missing,model=model,sglspan=sglspan,overlap=overlap)
        os.makedirs(cachedir,exist_ok=True)
        for s,crv in zip(missing,newcrvs):
            cachefn = cachefile(s)
            np.savez_compressed(cachefn,curvature=crv,meta=np.array(json.dumps(metadata(s))))
            print(f'wrote curvature to {cachefn}')
            crvs[s] = crv

    vals = []
    for s in spans:
        crv = crvs[s]
        if angles:
            # clip: dot products of numerically-unit tangents can exceed 1 and yield nan
            crv = np.arccos(np.clip(crv,-1,1))
            if degrees:
                crv = np.rad2deg(crv)
        vals.append(crv)

    return vals[0] if single else vals


def window_means(
    crv: np.ndarray,
    avg_span: int = 2000,
    ) -> tuple[np.ndarray,np.ndarray]:
    """Mean of crv over consecutive non-overlapping windows of avg_span values.

    Returns the window edges, as indices into crv, and the mean within every window.
    There are len(edges)-1 means. The last window is shorter than avg_span whenever
    avg_span does not divide len(crv) evenly.
    """
    if avg_span < 1:
        raise ValueError(f'avg_span needs to be at least 1, encountered {avg_span}')
    edges = np.append(np.arange(0,len(crv),avg_span),len(crv))
    means = np.array([np.mean(crv[edges[i]:edges[i+1]]) for i in range(len(edges)-1)])
    return edges, means


if __name__ == '__main__':

    vls = pcg.gen_params('cgna+','ATCG')

    fn = 'Test/unmethylated_lambda'
    with open(fn,'r') as f:
        seqlamb = f.read().strip()


    sglspan = 5000
    overlap = 500

    # sp = 10000
    # fst = 10000

    # seq = seqlamb[fst:fst+sp]


    # generate random sequence
    span = [1,2,5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250]
    nbp = len(seqlamb)
    avg_span = 2500
    model = 'cgna+'
    angles = True
    degrees = True

    # seq = ''.join(['atcg'[np.random.randint(4)] for i in range(nbp)])
    # seq = 'ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT'

    crvs = dnacurv_cached(fn,span,model=model,angles=angles,degrees=degrees,sglspan=sglspan,overlap=overlap)

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    # Leave text as text in the SVG
    mpl.rcParams['svg.fonttype'] = 'none'
    # (Optional) choose a font you have installed:
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    def cm_to_inch(cm: float) -> float:
        return cm/2.54

    def set_xlim_to_data(ax):
        ax.set_xlim(np.nanmin([np.nanmin(l.get_xdata()) for l in ax.get_lines()] + [np.nanmin(c.get_offsets()[:,0]) for c in ax.collections if c.get_offsets().size]), 
                np.nanmax([np.nanmax(l.get_xdata()) for l in ax.get_lines()] + [np.nanmax(c.get_offsets()[:,0]) for c in ax.collections if c.get_offsets().size]))


    ##################################################
    # general Figure Setup
    axlinewidth = 0.8
    axcolor     ='grey'
    axalpha     = 0.7
    axtick_major_width  = 0.8
    axtick_major_length = 2.4
    axtick_minor_width  = 0.4
    axtick_minor_length = 1.6

    tick_pad        = 2
    tick_labelsize  = 5
    label_fontsize  = 6
    legend_fontsize = 6

    panel_label_fontsize = 8
    label_fontweight= 'bold'
    panel_label_fontweight= 'bold'

    fig_width = 8.6
    fig_height = 5

    ##################################################
    # Main

    for plotspan in span:

        crv = crvs[span.index(plotspan)] if isinstance(span,list) else crvs
        fig = plt.figure(figsize=(cm_to_inch(fig_width), cm_to_inch(fig_height)), dpi=300,facecolor='w',edgecolor='k') 
        axes = []
        axes.append(plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1,rowspan=1))
        ax1 = axes[0]

        # mean curvature over windows of avg_span reference base pairs, drawn behind
        # the trace. align='edge' with the width taken from the edges keeps the last,
        # possibly shorter, window at its true extent.
        edges, means = window_means(crv,avg_span)
        ax1.bar(edges[:-1],means,width=np.diff(edges),align='edge',
                color='#bcd4ec',edgecolor='#2f6da8',linewidth=0.7,zorder=1)

        ax1.plot(np.arange(len(crv)),crv,color='black',linewidth=0.5,alpha=0.3,zorder=2)

        scatter_marker = 'o'
        scatter_linewidth = 0.8
        scatter_alpha = 0.8
        scatter_size = 20
        scatter_color = 'blue'
        # ax1.scatter(np.arange(len(crv)),crv,color=scatter_color,s=scatter_size,edgecolor='black',linewidth=scatter_linewidth,marker=scatter_marker,alpha=scatter_alpha)

        set_xlim_to_data(ax1)

        xlabel_pos = [0.5,-.07]
        ylabel_pos = [-0.04,0.5]

        ax1.set_xlabel('Reference Base Pair',fontsize=label_fontsize,fontweight=label_fontweight)
        ax1.xaxis.set_label_coords(*xlabel_pos)

        ax1.set_ylabel(r'Segment Spanning Bend Angle (deg)',fontsize=label_fontsize,fontweight=label_fontweight)
        ax1.yaxis.set_label_coords(*ylabel_pos)


        for ax in axes:
            
            ###############################
            # set major and minor ticks
            # ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,)
            # ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length)
            ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,color='#cccccc')
            ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length,color='#cccccc')

            ###############################
            ax.xaxis.set_ticks_position('both')
            # set ticks right and top
            ax.yaxis.set_ticks_position('both')
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(axlinewidth)
                ax.spines[axis].set_color(axcolor)
                ax.spines[axis].set_alpha(axalpha)

        ##############################################
        # Setup subpanels

        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.11,
            top=0.97,
            wspace=0.25,
            hspace=0.25
            )


        ##################################################
        # Save Figure
        savefn = f'{fn}_curvature_span_{plotspan}'

        fig.savefig(savefn+'.pdf',dpi=300,transparent=True)
        # fig.savefig(savefn+'.svg',dpi=300,transparent=True)
        fig.savefig(savefn+'.png',dpi=300,transparent=False)

        np.save(savefn+'.npy',crv)

        plt.close(fig)

