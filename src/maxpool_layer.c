#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    
    int col, row,c,n;
    int count=0;
    int length = (l.size-1)/2;
    if(l.size%2==0){
    for(n=0;n<in.rows;n++){
        for(c=0;c<l.channels; c++){
            for(row=0; row < l.height; row+= l.stride){
                for(col=0; col<l.width; col+= l.stride){
                    int window_x, window_p, num,cur;
                    float max = -999999.0;
                    cur=n*l.width*l.height*l.channels+c*l.height*l.width+row*l.width;
                    for(num=0; num < l.size; num++){
                        for(window_x=col; window_x < col+l.size; window_x++){
                            window_p = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+window_x;
                            int bound_w_end= n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+l.width;
                            int bound_w_start = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width;
                            float val;
                            if(window_p<n*l.height*l.width*l.channels+c*l.height*l.width||
                            window_p>=n*l.height*l.width*l.channels+(c+1)*l.height*l.width||
                            window_p >= bound_w_end||
                            window_p < bound_w_start){
                                continue;
                            }else{
                                val=in.data[window_p];
                            }
                             if(val>max) max=val;
                        }
                    }
                    out.data[count++]=max;
                }
            }
        }
    }
    }else{
        for(n=0;n<in.rows;n++){
        for(c=0;c<l.channels; c++){
            for(row=0; row < l.height; row+= l.stride){
                for(col=0; col<l.width; col+= l.stride){
                    int window_x, window_p, num,cur;
                    float max = -999999.0;
                    for(num=-length; num <= length; num++){
                        for(window_x=col-length; window_x <= col+length; window_x++){
                            window_p = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+window_x;
                            int bound_w_end= n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+l.width;
                            int bound_w_start = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width;
                            float val;
                            if(window_p<n*l.height*l.width*l.channels+c*l.height*l.width||
                            window_p>=n*l.height*l.width*l.channels+(c+1)*l.height*l.width||
                            window_p >= bound_w_end||
                            window_p < bound_w_start){
                                continue;
                            }else{
                                val=in.data[window_p];
                            }
                             if(val>max) max=val;
                        }
                    }
                    out.data[count++]=max;
                }
            }
        }
    }
    }
    //printf("%in out", l.height);


    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    int c_index, col, row,c,n;
    int count=0;
    int length = (l.size-1)/2;
    if(l.size%2==0){
    for(n=0;n<in.rows;n++){
        for(c=0;c<l.channels; c++){
            for(row=0; row < l.height; row+= l.stride){
                for(col=0; col<l.width; col+= l.stride){
                    int window_x, window_p, num,cur;
                    float max = -999999.0;
                    int index=-1;
                    cur=n*l.width*l.height*l.channels+c*l.height*l.width+row*l.width;
                    for(num=0; num < l.size; num++){
                        for(window_x=col; window_x < col+l.size; window_x++){
                            window_p = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+window_x;
                            int bound_w_end= n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+l.width;
                            int bound_w_start = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width;
                            float val;
                            if(window_p<n*l.height*l.width*l.channels+c*l.height*l.width||
                            window_p>=n*l.height*l.width*l.channels+(c+1)*l.height*l.width||
                            window_p >= bound_w_end||
                            window_p < bound_w_start){
                                continue;
                            }else{
                                val=in.data[window_p];
                            }
                             if(val>max) {
                                 max=val;
                                 index=window_p;
                             }
                        }
                    }
                    dx.data[index]+=dy.data[count++];
                }
            }
        }
    }
    }else{
        for(n=0;n<in.rows;n++){
        for(c=0;c<l.channels; c++){
            for(row=0; row < l.height; row+= l.stride){
                for(col=0; col<l.width; col+= l.stride){
                    int window_x, window_p, num,cur;
                    float max = -999999.0;
                    int index=-1;
                    cur=n*l.width*l.height*l.channels+c*l.height*l.width+row*l.width;
                    for(num=-length; num <= length; num++){
                        for(window_x=col-length; window_x <= col+length; window_x++){
                            window_p = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+window_x;
                            int bound_w_end= n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width+l.width;
                            int bound_w_start = n*l.height*l.width*l.channels+c*l.height*l.width+row*l.width+num*l.width;
                            float val;
                           if(window_p<n*l.height*l.width*l.channels+c*l.height*l.width||
                            window_p>=n*l.height*l.width*l.channels+(c+1)*l.height*l.width||
                            window_p >= bound_w_end||
                            window_p < bound_w_start){
                                continue;
                            }else{
                                val=in.data[window_p];
                            }
                             if(val>max){
                                 max=val;
                                 index=window_p;
                             } 
                        }
                    }
                    dx.data[index]+=dy.data[count++];
                }
            }
        }
    }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

