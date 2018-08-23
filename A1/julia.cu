#include <stdio.h>

#define DIM 1000


struct cppComplex {
	float r; 
	float i;
	cppComplex( float a, float b ) : r(a), i(b) {}
	float magnitude2( void ) {
		return r * r + i * i;
	}
	cppComplex operator*(const cppComplex& a) {
		return cppComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cppComplex operator+(const cppComplex& a) {
		return cppComplex(r+a.r, i+a.i);
	}
};

int julia_cpu( int x, int y ) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	
	cppComplex c(-0.8, 0.156);
	cppComplex a(jx, jy);

	int i = 0;
	for(i=0; i<200; i++){
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

void julia_set_cpu() {

	unsigned char *pixels = new unsigned char[DIM * DIM]; 

	for (int x = 0; x < DIM; ++x) {
		for (int y = 0; y < DIM; ++y) {
			pixels[x + y * DIM] = 255 * julia_cpu(x, y);
		}
	}

	FILE *f = fopen("julia_cpu.ppm", "wb");

    fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
    
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            fputc(pixels[(y * DIM + x)], f);
            fputc(0, f);
            fputc(0, f);
      }
    }
    fclose(f);

    delete [] pixels;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
/*Begin the GPU part*/
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////



__global__ void kernel( unsigned char *ptr ) {
	/*
		wirte your kernel code here
	*/
}

void julia_set_gpu() {

	unsigned char *pixels = new unsigned char[DIM * DIM]; 

	/*
		wirte the host code here
	*/
	
	//kernel<<<grid,1>>>(dev_bitmap);

	/*
		write the code to copy the data back
	*/

	FILE *f = fopen("julia_gpu.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            fputc(pixels[(y * DIM + x)], f);   // 0 .. 255
            fputc(0, f);
            fputc(0, f);
      }
    }


    fclose(f);

	delete [] pixels; 

}



int main( void ) {
	
	julia_set_cpu();
	julia_set_gpu();

}
