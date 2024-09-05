#!/usr/bin/gnuplot
plot "log.dat" u 1:4 ti "kinetic" w l, "" u 1:5 ti "potential" w l, "" u 1:6 ti "contact" w l, \
     "" u 1:7 ti "fracture" w l,  "" u 1:($4+$5+$6+$7) w l lw 3 ti "kin + post + contact"
pause -1
