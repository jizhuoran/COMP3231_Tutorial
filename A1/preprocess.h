typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "COMP3231"
#define RGB_COMPONENT_COLOR 255

static PPMImage *readPPM(const char *filename);
void writePPM(const char *filename, PPMImage *img);