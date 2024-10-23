import monai.transforms as transforms
from datetime import datetime

class UsgTransforms:
    def __init__(self) -> None:
        pass
    
    def log(self, payload):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp, "LOG from ",self.__class__.__name__, ":" ,payload)

    def a_t_default(self):
        return [
            # # MUST BE 
            transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3),
            transforms.RandAdjustContrast(prob=0.5, gamma=(0.5, 4.5)),
            transforms.RandHistogramShift(num_control_points=10, prob=0.5),
            transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5),

            transforms.RandFlip(prob=0.5, spatial_axis=1),

            transforms.RandGibbsNoise(prob=0.5, alpha=(0.4, 1.0)),
            
            transforms.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.4),
            transforms.RandRotate(range_x=10.0, prob=0.5, keep_size=True),
            
            transforms.RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5),prob=0.8),
                                        
            transforms.RandAffine(prob=0.5),
            transforms.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.03, 0.03)),
            transforms.Rand2DElastic(spacing=(20,20), magnitude_range=(1,2), prob=0.4),
            transforms.RandCoarseDropout(holes=5, spatial_size=(30,30), fill_value=0,prob=0.6),
            transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3)
        ]

    def a_t_without_intensity(self):
        return [
            # # MUST BE 
            transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3),
            # transforms.RandAdjustContrast(prob=0.5, gamma=(0.5, 4.5)),
            # transforms.RandHistogramShift(num_control_points=10, prob=0.5),
            transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5),

            transforms.RandFlip(prob=0.5, spatial_axis=1),

            transforms.RandGibbsNoise(prob=0.5, alpha=(0.4, 1.0)),
            
            transforms.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.4),
            transforms.RandRotate(range_x=10.0, prob=0.5, keep_size=True),
            
            transforms.RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5),prob=0.8),
                                        
            transforms.RandAffine(prob=0.5),
            transforms.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.03, 0.03)),
            transforms.Rand2DElastic(spacing=(20,20), magnitude_range=(1,2), prob=0.4),
            transforms.RandCoarseDropout(holes=5, spatial_size=(30,30), fill_value=0,prob=0.6),
            transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3)
        ]
    
    def a_t_without_noises(self):
        return [
            # # MUST BE 
            # transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3),
            transforms.RandAdjustContrast(prob=0.5, gamma=(0.5, 4.5)),
            transforms.RandHistogramShift(num_control_points=10, prob=0.5),
            transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5),

            transforms.RandFlip(prob=0.5, spatial_axis=1),

            # transforms.RandGibbsNoise(prob=0.5, alpha=(0.4, 1.0)),
            
            transforms.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.4),
            transforms.RandRotate(range_x=10.0, prob=0.5, keep_size=True),
            
            # transforms.RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5),prob=0.8),
                                        
            transforms.RandAffine(prob=0.5),
            transforms.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.03, 0.03)),
            transforms.Rand2DElastic(spacing=(20,20), magnitude_range=(1,2), prob=0.4),
            transforms.RandCoarseDropout(holes=5, spatial_size=(30,30), fill_value=0,prob=0.6),
            # transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3)
        ]
    
    def a_t_only_basic(self):
        return [
            # # MUST BE 
            # transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3),
            # transforms.RandAdjustContrast(prob=0.5, gamma=(0.5, 4.5)),
            # transforms.RandHistogramShift(num_control_points=10, prob=0.5),
            # transforms.RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.5),

            transforms.RandFlip(prob=0.5, spatial_axis=1),

            # transforms.RandGibbsNoise(prob=0.5, alpha=(0.4, 1.0)),
            
            transforms.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.4),
            transforms.RandRotate(range_x=10.0, prob=0.5, keep_size=True),
            
            # transforms.RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5),prob=0.8),
                                        
            transforms.RandAffine(prob=0.5),
            transforms.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.03, 0.03)),
            transforms.Rand2DElastic(spacing=(20,20), magnitude_range=(1,2), prob=0.4),
            transforms.RandCoarseDropout(holes=5, spatial_size=(30,30), fill_value=0,prob=0.6),
            # transforms.RandGaussianNoise(prob=0.5, mean=0.3, std=0.3)
        ]
    
    def empty_transforms(self):
        return []


