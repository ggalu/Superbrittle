#!/usr/bin/gnuplot
# Plot of file.dat 

# This command works for a linux computer. In linux, you need to specify the exact location of
# the font you want to use


# nomirror means do not put tics on the opposite side of the plot
set xtics nomirror
set ytics nomirror

# On the Y axis put a major tick every 5
#set ytics 5

# On both the x and y axes split each space in half and put a minor tic there
#set mxtics 2
#set mytics 2


# Line style for axes
# Define a line style (we're calling it 80) and set 
# lt = linetype to 0 (dashed line)
# lc = linecolor to a gray defined by that number
set style line 80 lt 0 lc rgb "#808080"

# Set the border using the linestyle 80 that we defined
# 3 = 1 + 2 (1 = plot the bottom line and 2 = plot the left line)
# back means the border should be behind anything else drawn
set border 3 back ls 80 

# Line style for grid
# Define a new linestyle (81)
# linetype = 0 (dashed line)
# linecolor = gray
# lw = lineweight, make it half as wide as the axes lines
set style line 81 lt 0 lc rgb "#808080" lw 0.5

# Draw the grid lines for both the major and minor tics
set grid xtics
set grid ytics
set grid mxtics
set grid mytics

# Put the grid behind anything drawn and use the linestyle 81
set grid back ls 81

# Add line at -3db
# Draw a line from the right end of the graph to the left end of the graph at
# the y value of -3
# The line should not have an arrowhead
# Linewidth = 2
# Linecolor = black
# It should be in front of anything else drawn
#set arrow from graph 0,first -3 to graph 1, first -3 nohead lw 2 lc rgb "#000000" front

# Put a label -3db at 80% the width of the graph and y = -2 (it will be just above the line drawn)
#set label "-3dB" at graph 0.8, first -2

# Create some linestyles for our data
# pt = point type (triangles, circles, squares, etc.)
# ps = point size
set style line 1 lt 1 lc rgb "#A00000" lw 2 #pt 7 ps 0
set style line 2 lt 1 lc rgb "#00A000" lw 2 pt 11 ps 1.5
set style line 3 lt 1 lc rgb "#5060D0" lw 2 pt 9 ps 1.5
set style line 4 lt 1 lc rgb "#0000A0" lw 2 pt 8 ps 1.5
set style line 5 lt 1 lc rgb "#D0D000" lw 2 pt 13 ps 1.5
set style line 6 lt 1 lc rgb "#00D0D0" lw 2 pt 12 ps 1.5
set style line 7 lt 1 lc rgb "#B200B2" lw 2 pt 5 ps 1.5



# Put X and Y labels
set xlabel "time / s"
set ylabel "momentum, kg.m/s"

# Set the range of our x and y axes
#set xrange [1:10]
#set yrange [-30:5]

# Give the plot a title
set title "momentum conservation"

# Put the legend at the bottom left of the plot
#set key left bottom
set nokey



set yrange [0:]

plot "log.dat" u 1:3 w l ls 1

pause -1

set terminal png notransparent rounded giant font "/usr/share/fonts/msttcore/arial.ttf" 24 size 1200,960 
set output "tf.png"
plot "log.dat" u 1:3 w l ls 1