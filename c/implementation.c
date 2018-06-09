//
//wolfgang.c
//
//
//Created by Olaf Flebbe on 25.05 .18.
//

//include "wolfgang.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define TAU 800
#define DURATION 5
#define RINGSIZE 50
#define RINGS 200
#define N (DURATION*50)

float		X        [TAU][RINGS][RINGSIZE];
float		Y        [TAU][RINGS][RINGSIZE];

double		VM2 = 20.0;
////10 ^ -6 M / s
double		VM3 = 23.0;
//10 ^ -6 M / s
double		K2 = 1.0;
//10 ^ -6 M / s
double		KR = 0.8;
//10 ^ -6 M / s
double		KA = 0.9;
//10 ^ -6 M / s
double		kf = 1.0;
//1 / s
double		k = 0.8;
//1 / s
double		v = 0.325;
//v = v0 + v1 * beta


double		Dxl = 2.0;
//1 / s
double		Dxr = 6.0;
//1 / s
double		Dxd = 1.0;
//1 / s
double		kt = 0.1;
//1 / s

double		sampleFreq = 1.0;
//s
int		duration = 500;

double		omega = 1.047;
double		Am = 1.2;

double		h = 0.02;

//int           (duration / h) + 1;


float		X0       [RINGSIZE][RINGS];
float		Y0       [RINGSIZE][RINGS];

float cylinderConcPerCell[RINGSIZE][N];
float totalConcVector[2][N];
float totalDiffConcVector[RINGSIZE][N];


double
f(double x, double y)
{
	return VM2 * x / (K2 + x) - VM3 * y * (x * x) / ((KR + y) * (KA * KA + x * x)) \
	-kf * y;
}

double
dXdt(double x, double y, double flux)
{
	return v - f(x, y) - k * x + flux;
}

double
dYdt(double x, double y)
{
	return f(x, y);
}

int
main()
{
    int filecounter = 0;
    FILE *outf;
    
	for (int t = 0; t < TAU; t++) {
		for (int ring = 0; ring < RINGS; ring++) {
			for (int cell = 0; cell < RINGSIZE; cell++) {
				X[t][ring][cell] = 0.;
                                Y[t][ring][cell] = 0.;
			}
		}
	}
    for (int t = 0; t < N; t++) {
        for (int cell = 0; cell < RINGSIZE; cell++) {
            cylinderConcPerCell[cell][ t] = 0.;
        }
    }
	for (int ring = 0; ring < RINGS; ring++) {
		for (int cell = 0; cell < RINGSIZE; cell++) {
			X0[cell][ring] = (0 - 0.5) * 0.1 + 0.406;
			Y0[cell][ring] = (0 - 0.5) * 0.2 + 2.76;
		}
	}

    
        for (int ring = 0; ring < RINGS; ring++) {
            for (int cell = 0; cell < RINGSIZE; cell++) {
                X[0][ring][cell] = X0[cell][ring];
                Y[0][ring][cell] = Y0[cell][ring];
            }
        }
  

	for (int t = 0; t < N; t++) {
        char buf[256];
        int		index = t % TAU;
        double x, y;
        double angle = 2*M_PI/RINGSIZE;
        
		for (int ring = 0; ring < RINGS; ring++) {
			for (int cell = 0; cell < RINGSIZE; cell++) {
				double		flux = 0.;
				/* Sinus - Exitation */
				if (ring <= 3 && 24 <= cell && cell <= 27)
					X[index][ring][cell] = Am * 0.5 * (1 + cos(omega * t * h));
				/* horizontal diffusion */
                if (cell == 0) {
					flux = Dxr * (X[index][ring][RINGSIZE - 1] - X[index][ring][cell])
						+ Dxl * (X[index][ring][1] - X[index][ring][cell]);
                } else if (cell == RINGSIZE - 1) {
					flux = Dxr * (X[index][ring][cell - 1] - X[index][ring][cell])
						+ Dxl * (X[index][ring][0] - X[index][ring][cell]);
                } else {
					flux = Dxr * (X[index][ring][cell - 1] - X[index][ring][cell])
						+ Dxl * (X[index][ring][cell + 1] - X[index][ring][cell]);
                }
				/* vertical diffusion */
				if (ring == 0) {
					if (X[index][ring][cell] > X[index][ring + 1][cell])
						flux -= Dxd * (X[index][ring][cell] - X[index][ring + 1][cell]);
				} else if (ring == RINGS - 1) {
					if (X[index][ring - 1][cell] > X[index][ring][cell])
						flux += Dxd * (X[index][ring - 1][cell] - X[index][ring][cell]);
				} else {
					if (X[index][ring - 1][cell] > X[index][ring][cell])
						flux += Dxd * (X[index][ring - 1][cell] - X[index][ring][cell]);
					if (X[index][ring][cell] > X[index][ring + 1][cell])
						flux -= Dxd * (X[index][ring][cell] - X[index][ring + 1][cell]);
				}
				/*
				 * Feedback across ring, tau elements back in
				 * time
				 */
				if (t >= TAU)
					flux += kt * (X[(t - TAU) % TAU][ring][(cell + RINGSIZE / 2) % RINGSIZE] - X[index][ring][cell]);

				float		x = X[index][ring][cell];
				float		y = Y[index][ring][cell];
				double		k11 = h * dXdt(x, y, flux);
				double		k12 = h * dYdt(x, y);

				double		k21 = h * dXdt(x + 0.5 * k11, y + 0.5 * k12, flux);
				double		k22 = h * dYdt(x + 0.5 * k11, y + 0.5 * k12);

				double		k31 = h * dXdt(x + 0.5 * k21, y + 0.5 * k22, flux);
				double		k32 = h * dYdt(x + 0.5 * k21, y + 0.5 * k22);

				double		k41 = h * dXdt(x + k31, y + k32, flux);
				double		k42 = h * dYdt(x + k31, y + k32);
				X[(index + 1) % TAU][ring][cell] = x + (k11 + 2 * k21 + 2 * k31 + k41) / 6;
				Y[(index + 1) % TAU][ring][cell] = y + (k12 + 2 * k22 + 2 * k32 + k42) / 6;
                
                cylinderConcPerCell[cell][t] += X[index][ring][cell];
			}
            
        }
       
        if (t%50 == 0 ) {
            printf("fertig\n");
            sprintf( buf, "/tmp/pic%02d.txt", filecounter++ );
            outf = fopen( buf, "w");
            for (int ring = 0; ring < RINGS; ring++) {
                for (int cell = 0; cell < RINGSIZE; cell++) {
                    fprintf( outf, "%lf ", X[(index + 1) % TAU][ring][cell]);
                }
                fprintf(outf, "\n");
            }
            fclose(outf);
        }
        x = y = 0.;
        for (int cell = 0; cell < RINGSIZE; cell++) {
            x += cylinderConcPerCell[cell][t]*cos(angle*cell);
            y += cylinderConcPerCell[cell][t]*sin(angle*cell);
        }
// Angle
        totalConcVector[0][t] = atan(y/x);
        if (totalConcVector[0][t]<0) {
            totalConcVector[0][t] += 2*M_PI;
        }
        totalConcVector[0][t] = totalConcVector[0][t]/(2*M_PI)*RINGSIZE;
// Length
        totalConcVector[1][t] = sqrt(x*x+y*y);
    }
    outf = fopen( "/tmp/cylcon.txt", "w");
    for (int cell= 0; cell < RINGSIZE; cell++) {
        for (int t = 0; t < N; t++) {
                fprintf( outf, "%lf ", cylinderConcPerCell[cell][t]);
        }
        fprintf(outf, "\n");
    }
    fclose(outf);
                
// Calculate total diff concentration vector across cylinderConcPerCell
    for (int cell= 0; cell < RINGSIZE; cell++) {
        for (int t = 0; t < N; t++) {
            totalDiffConcVector[cell][t] = cylinderConcPerCell[cell][t] - cylinderConcPerCell[(cell + (RINGSIZE/2))%RINGSIZE][t];
        }
    }
    
    outf = fopen( "/tmp/total.txt", "w");
        for (int cell= 0; cell < RINGSIZE; cell++) {
            for (int t = 0; t < N; t++) {
                fprintf( outf, "%lf ", totalDiffConcVector[cell][t]);
        }
        fprintf(outf, "\n");
    }
    fclose(outf);
}
