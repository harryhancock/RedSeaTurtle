from wildlife_datasets import analysis, datasets, loader

datasets.SeaTurtleID2022.get_data('data/SeaTurtleID2022')
d = datasets.MacaqueFaces('data/SeaTurtleID2022')
d.df
d.plot_grid()
analysis.display_statistics(d.df)
d.summary