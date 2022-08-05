# fusion_depth_mvs
This repo aims to fuse RGB-D data from sensor or MVS pipelines.

## Data Format

### DTU
You can test our code in [DTU](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) dataset, where the link is from [MVSNet](https://github.com/YoYo000/MVSNet).

Each project folder should contain the following
```
.                          
├── images                 
│   ├── 00000000.jpg       
│   ├── 00000001.jpg       
│   └── ...                
├── cams                   
│   ├── 00000000_cam.txt   
│   ├── 00000001_cam.txt   
│   └── ...                
└── pair.txt               
```

### Demo RGB-D Data
You can download the demo data at [demo](https://drive.google.com/file/d/1wRDZq8DVsCTHvC_MVy0_VlFfoD8oxFLX/view?usp=sharing), which is taken from the iPad-Pro.

## Useage
```shell
# $1 for the data directory, $2 for the output .ply name, $3 for the fusion method.
python3 fuse/fuse_depth.py --dense_folder $1 --name $2 --fusion mvs

# Demo
python3 fuse/fuse_depth.py --dense_folder dtu/scan1 --name scan1 --fusion mvs
```

## TODO
- [x] Upload demo data and expected result.
- [ ] Visualization for .pfm
- [ ] Support RGB-D sequence from sensor like Intel RealSense.

## Acknowledgements
Thanks to Gu Xiaodong's [CascadeMVSNet](https://github.com/alibaba/cascade-stereo) and xy-guo's [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch), the fusion script highly refers its testing code.