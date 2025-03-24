sizex = 150;
sizey = 30;
sizez = 10;

nx = 20;
ny = 10;
nz = 3;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {sizex, 0, 0, 1.0};
Point(3) = {sizex, sizey, 0, 1.0};
Point(4) = {0, sizey, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {3, 4, 1, 2};
Plane Surface(1) = {1};

Transfinite Surface {1};
Transfinite Curve {1, 3} = nx+1 Using Progression 1;
Transfinite Curve {2, 4} = ny+1 Using Progression 1;
Recombine Surface {1};

Extrude {0, 0, sizez} {Surface{1}; Layers{nz}; Recombine; }
