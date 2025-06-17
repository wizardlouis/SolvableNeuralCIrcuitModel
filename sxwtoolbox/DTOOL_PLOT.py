import matplotlib.pyplot as plt
from .DTOOL_OS import opj

def pf(a1, a2): plt.figure(figsize=(a1, a2))

def pfsp(m, n, a1, a2):
    fig, ax = plt.subplots(m, n, figsize=(a1, a2))
    return fig, ax

def axe_customize(
    ax,
    visible={
        'left':True,
        'right':False,
        'top':False,
        'bottom':True
    },
    linewidth=3,
    tickwidth=3,
):
    for key,value in zip(visible.keys(),visible.values()):
        ax.spines[key].set_visible(value)
        ax.spines[key].set_linewidth(linewidth)
    ax.tick_params(width=tickwidth)

def save_fig(
        fig,save_dir,filename,format='png'
):
    PIXEL_IMAGE_NAME=[
        'jpg','jpeg',
        'png',
        'tiff'
    ]
    VECTOR_IMAGE_NAME=[
        'svg',
        'pdf',
        'eps'
    ]
    TRANSPARENT_IMAGE_NAME=[
        'png',
        'pdf',
        'svg',
        'tiff'
    ]
    assert (format in PIXEL_IMAGE_NAME) or (format in VECTOR_IMAGE_NAME)
    FIG_KWARGS=dict(
        bbox_inches='tight'
    )
    if format in TRANSPARENT_IMAGE_NAME:
        FIG_KWARGS['transparent']=True
    fig_path=opj(
        save_dir,f"{filename}.{format}"
    )
    fig.savefig(
        fig_path
    )