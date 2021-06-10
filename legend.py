import matplotlib.pyplot as plt

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
Tech = ['Nuclear','Solar','SolarDSSC','Hydro','SolarUtility','SolarUtilityDSSC','WindOffshore','WindOnshore','PumpedStorage','Coal','Gas']
legend = plt.legend(labels=Tech, loc=3, framealpha=1, frameon=False)

def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
plt.show()
