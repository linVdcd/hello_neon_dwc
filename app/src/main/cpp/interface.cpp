//
// Created by lin on 17-8-4.
//
#include "interface.h"
#include "dw.h"
#include "types.h"
#include <vector>

void SetNeonDimStrides(Dims<4>* d) {
    long int stride = 1;
    for (int i = 0; i < 4; ++i) {
        d->strides[i] = stride;
        stride *= d->sizes[i];
    }
}
void toDims(Dims<4>* d,int a[]){
    for (int i =0;i<4;i++)
        d->sizes[i]=a[i];
}

void dw_interface(){
    int batch=1;
    float input_data[48];
    int a[4] ={3,4,4,batch};
    Dims<4> input_dims; toDims(&input_dims,a);SetNeonDimStrides(&input_dims);
    float filter_data[27];
    int b[4] ={3,3,3,1};
    Dims<4> filter_dims; toDims(&filter_dims,b);SetNeonDimStrides(&filter_dims);
    float  bias_data[3]={0,0,0};
    int c[4] = {3,1,1,1};
    Dims<4> bias_dims; toDims(&bias_dims,c);SetNeonDimStrides(&bias_dims);
    int stride=1;
    int pad_width=1;
    int pad_height=1;
    int depth_multiplier=1;
    float output_data[48];
    int d[4]={3,4,4,batch};
    Dims<4> output_dims; toDims(&output_dims,d); SetNeonDimStrides(&output_dims);
    float aa[1][4][4][3];
    //input_data = (float*)malloc(sizeof(float)*batch*4*4*3);
    //filter_data = (float*)malloc(sizeof(float)*27);
    //output_data = (float*)malloc(sizeof(float)*4*4*3*batch);
    int batchSize = 4*4*3;

    for(int b=0;b<batch;b++)
        for (int h = 0;h<4;h++)
            for(int w =0;w<4;w++)
                for (int c =0;c<3;c++)
                    input_data[b*batchSize+h*4*3+w*3+c]= float(c)+(float)1.0;

    for (int h=0;h<3;h++)
        for(int w=0;w<3;w++)
            for(int c=0;c<3;c++)
                filter_data[h*3*3+w*3+c] = float(c)+float(1.0);





    DepthwiseConv<FusedActivationFunctionType::kNone>(input_data, input_dims,
                                                      filter_data, filter_dims,
                                                      bias_data, bias_dims, stride,
                                                      pad_width, pad_height, depth_multiplier,
                                                      output_data, output_dims);


    for(int b=0;b<batch;b++)
        for (int h = 0;h<4;h++)
            for(int w =0;w<4;w++)
                for (int c =0;c<3;c++)
                    aa[b][h][w][c]=output_data[b*batchSize+h*4*3+w*3+c];
}