#include<stdio.h>
#include<stdlib.h>

#include "preprocess.h"


void changeColorPPM(PPMImage *img)
{
    int i;
    if(img){

         for(i=0;i<img->x*img->y;i++){
              img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
              img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
              img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
         }
    }
}

int main(){
    PPMImage *image;
    image = readPPM("input.ppm");

    changeColorPPM(image);
    your_gaussian_blur_func(image);

    writePPM("output.ppm",image);

}