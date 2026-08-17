/*
 * GraphX: Plot integration with the project's matplotlib/seaborn stack.
 *
 * Provides NetworkX-style network visualization that lives on top of the
 * existing plot/chart API:
 *  - draw_networkx_nodes    → rendered as scatter points on a BaseChart
 *  - draw_networkx_edges    → rendered as reference line segments
 *  - draw_networkx_labels   → rendered as text annotations
 *  - draw_networkx_edge_labels → rendered as text annotations on edges
 *  - draw_networkx          → convenience combo (mirrors nx.draw / nx.draw_networkx)
 *
 * The output is a {@link org.bytedeco.pytorch.plot.chart.BaseChart} subclass
 * that can be rendered to PNG/JPEG and combined with any other plot (line,
 * scatter, heatmap, bar) using the standard matplotlib {@code subplots} idiom.
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx) draw_networkx
 * functions. BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.plot;