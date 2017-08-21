Fixed bug with first plot in pairwise_cluster_comparison
Fixed numerical issue bug with t-test
Accounted for edge case when num_genes is super high for pairwise_cluster_comparison
For plot_label_legends, added an option to show axes. Also added an option for log scales. Also added an option for legend location.
For filter_genes, changed the print statement slightly
For sk_pca, added a line that centers the data first
For split.visualize_history, added a figure to each plot showing the distribution of expressions.
Added a comment in select_genes_using_Welchs
Made sure plot_label_legends accounts for the string '-1' as singletons
Added markersize option to plot_labels_legend. also changed singleton marker shape to white stars
Changed split.visualize_history and utils.plot_labels_legend to allow visualization with less samples
Added option to plot_labels_legend where setting legend_pos to None results in no legend shown
Added threshold option to split.visualize_history
Added option to visualize_history that allows plots to be saved separately
Added option to select_genes_using_welchs that returns t statistics and p values (also an option for equal variance)
