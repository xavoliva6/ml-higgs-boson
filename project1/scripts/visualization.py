import matplotlib.pyplot as plt


# display correlation dots in the paired plot
def corr_dot(*args, **kwargs):
    """
    Calculate the correlation of features and display it:
    positive correlation displayed with red colours,
    negative correlation with blue colours.
    The greater the absolute value of the correlation, 
    the greater the size of the corresponding number.
    """
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5, ], xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)
