from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
import pandas as pd
from scipy import ndimage, spatial


class RasterClipper:
    """Clips a raster to a provided boundary"""
    def __init__(self, raster_path, boundary, out_dir):
        self.raster_path = raster_path
        self.boundary = boundary
        self.out_dir = out_dir

        self.suffix = "clip"
        self.out_name = f"{self.raster_path.stem}_{self.suffix}.tif"
        self.out_path = Path(out_dir) / self.out_name

    def clip_raster(self):
        """Clip the raster to the provided boundaries"""

        # rasterio.mask.mask requires an iterable of shapely polygon(s)
        # If self.boundary is not iterable, cast to single element list
        if not hasattr(self.boundary, "__iter__"):
            self.boundary = [self.boundary]

        with rasterio.open(self.raster_path) as src:
            out_img, out_transform = rasterio.mask.mask(src, self.boundary, nodata=np.nan, crop=True)
            out_meta = src.meta

            out_meta.update({"driver": "GTiff",
                            "height": out_img.shape[1],
                            "width": out_img.shape[2],
                            "transform": out_transform,
                            "nodata": np.nan})
        
        return out_img, out_meta
    
    def execute(self):
        """Clip the raster to the provided geometry and write to disk"""
        out_img, out_meta = self.clip_raster()

        # Write to disk
        with rasterio.open(self.out_path, 'w', **out_meta) as dst:
            dst.write(out_img)
        
        return self.out_path
    

class BinaryClassifier:
    """Create a binary classification for a raster based on a certain threshold value"""

    def __init__(self, raster_path, threshold, out_dir) -> None:
        self.raster_path = raster_path
        self.threshold = threshold
        self.out_dir = out_dir

        self.suffix = "bin"
        self.out_name = f"{self.raster_path.stem}_{self.suffix}.tif"
        self.out_path = Path(out_dir) / self.out_name

    def bin_class(self, a):
        """
        Classify an array:
            1: all values greater than the threshold values
            0: all values less than the threshold values
        """
        a_copy = a.copy()

        # Save location of nans
        mask = np.isnan(a_copy)

        # Apply threshold
        a_copy[a_copy <= self.threshold] = 0
        a_copy[a_copy > self.threshold] = 1

        # Redefine nans
        a_copy[mask] = np.nan

        return a_copy
    
    def execute(self):
        """Call other methods to classify the raster and write to disk"""

        # Open raster
        raster = rasterio.open(self.raster_path)
        profile = raster.profile
        array = raster.read(1)

        # Classify raster
        bin_array = self.bin_class(array)

        # Write to disk
        with rasterio.open(self.out_path, 'w', **profile) as dst:
            dst.write(bin_array, 1)

        return self.out_path
    

class RasterAgreementAssessor:
    """Assesses the agreement between two binary rasters"""

    def __init__(self, profc_path, rt_path, bend_id:str, out_dir) -> None:
        self.profc_path = profc_path
        self.rt_path = rt_path
        self.bend_id = bend_id
        self.out_dir = out_dir

        self.agreement_suffix = "agr"
        self.composite_suffix = "comp"
        self.agreement_out_name = f"{self.bend_id}_{self.agreement_suffix}.tif"
        self.composite_out_name = f"{self.bend_id}_{self.composite_suffix}.tif"
        self.agreement_out_path = self.out_dir / self.agreement_suffix / self.agreement_out_name
        self.composite_out_path = self.out_dir / self.composite_suffix / self.composite_out_name

    def calc_adjust(a1, a2, dim):
        """
        Determine adjustment for the given dimension
        Here adjustment is defined as the value added to the beginning and end of the given dimension
        """

        # Dict used to translate dimension to an axis to aggregate accross
        dim_dict = {'row':1, 'col':0}


        # Determine where each array has solid nans along the given dimension
        a1_nans = np.isnan(a1).all(axis=dim_dict.get(dim))
        a2_nans = np.isnan(a2).all(axis=dim_dict.get(dim))

        sa, la = sorted([a1_nans,a2_nans], key = lambda x: x.shape)

        if la[0] == sa[0]: # Both arrays match at the beginning, so buffering occurs at the end of the array
            adj = 0
        elif la[0]: # NaN buffer at the beginning of la only, so sa needs to be moved one over
            adj = 1
        else:         # This would imply that la has a 2x Nan buffer [... True, True] - unlikely
            print((sa, sa.shape), (la, la.shape))
            print('Something went wrong')

        return adj
    
    def adjust_rasters(self, a1, a2):
        """Adjust the dimensions of two arrays if they are not the same."""

        # Determine the smallest array that can hold both arrays
        mr = np.max([a1.shape[0], a2.shape[0]])
        mc = np.max([a1.shape[1], a2.shape[1]])
        na = np.zeros((mr, mc))*np.nan

        # Calculate the adjustments that need to be made for the array that is smaller in a given dimension
        ## Row adjustment
        adj_r = self.calc_adjust(a1, a2, 'row')

        ## Col adjustment
        adj_c = self.calc_adjust(a1, a2, 'col')

        # For both of the input arrays, determine which array needs to be adjusted in which dimension before adding it to the canvas na array
        # If an array is smaller in a given dimension, then it is assumed that it needs the adjustment
        new_arrays = []
        for g in [a1, a2]:

            # Empty array
            canvas = na.copy()

            # Is this array both smaller in the first and second dimensions?
            if (g.shape[0]<mr) and (g.shape[1]<mc):
                canvas[0+adj_r : g.shape[0]+adj_r, 0+adj_c : g.shape[1]+adj_c] = g
                new_arrays.append(canvas.copy())

            # Is this array only smaller in the first dimension?
            elif g.shape[0]<mr:
                canvas[0+adj_r : g.shape[0]+adj_r, 0: g.shape[1]] = g
                new_arrays.append(canvas.copy())

            # Is this array only smaller in the second dimension?
            elif g.shape[1]<mc:
                canvas[0 : g.shape[0], 0+adj_c : g.shape[1]+adj_c] = g
                new_arrays.append(canvas.copy())

            # This array must be larger in both dimensions, and therefore does not need adjustment
            else:
                new_arrays.append(g)

        return new_arrays
    
    def assess_agreement(self, profc, rt):
        """
        Assess the agreement of the rasters. This is done by redefining all rt values from 1 to 10, then adding the profc and rt rasters together.
        
        Because the RT foreground values were redefined to a value of 10, when individual
        pixels are added together the values of each digit of a cell comunicate how the
        different methods agreed (or not), as shown int the confusion matrix below

                          | PC Ridge (01)| Not PC Ridge (00)|
        ------------------|--------------|------------------|
        RT Ridge     (10) |      11      |        10        |
        ------------------|--------------|------------------|
        Not RT Ridge (00) |      01      |        00        |
        ------------------|--------------|------------------|


        self.adjust_rasters() will automatically adjust the dimensions of either raster if they were altered in classification.
        """

        # Redefine RT foreground values as 10 for comparison
        rt[rt==1] = 10

        # Add arrays together to form the composite array
        comp = profc+rt

        # Create agreement array & write to disk
        agr = comp.copy()
        agr[agr== 1] = 0 # redefine all disagreement as 0
        agr[agr==10] = 0 # redefine all disagreement as 0
        agr[agr==11] = 1 # Keep all positve agreement as 1

        return comp, agr
    
    def execute(self):
        """
        Execute other methods to generate agreement raster
        
        Workflow:
        1. Check dimensions of both rasters
            - alter to minimum shared footprint if dimensions are not equal
        2. Assess agreement between the two bianry rasters to return agreement and composite rasters
            - agreement raster: 
                - 1 where both rasters agree there is a ridge
                - 0 where they do not
            - composite raster:
                - 11 where both asters agree there is a ridge
                - 00 where both rasters agree there is a swale
                - 10 where only profc says there is a ridge
                - 01 where only rt says there is a ridge
        3. Write agreement and composite rasters to disk
        """

        # Open rasters, get arrays
        profc_raster = rasterio.open(self.profc_path)
        profile = profc_raster.profile
        profc = profc_raster.read(1)

        rt_raster = rasterio.open(self.rt_path)
        rt = rt_raster.read(1)

        # Check array dimensions, adjust if necesarry 
        if profc.size != rt.size:
            profc, rt = self.adjust_rasters(profc, rt)

            # Update the raster profile info if dimensions are changed
            profile["width"] = profc.shape[1]
            profile["height"] = profc.shape[0]
        
        # Assess agreement of binary rasters 
        composite, agreement = self.assess_agreement(profc = profc, rt = rt)


        # Write composite and agreement rasters to disk
        with rasterio.open(self.composite_out_path, "w", **profile) as dst:
            dst.write(composite, 1)
        
        with rasterio.open(self.agreement_out_path, "w", **profile) as dst:
            dst.write(agreement, 1)

        return self.agreement_out_path


class RasterDenoiser:
    """Denoise a binary raster by removing small image objects from both foreground (1s) and background (0s)"""

    def __init__(self, raster_path, small_feats_size, elongation_threshold, out_dir) -> None:
        self.raster_path = raster_path
        self.small_feats_size = small_feats_size
        self.elongation_threshold = elongation_threshold
        self.out_dir = out_dir

        self.suffix = "denoise"
        self.out_name = f"{self.raster_path.stem}_{self.suffix}{self.small_feats_size}px{self.elongation_threshold}p.tif"
        self.out_path = self.out_dir / self.out_name

    def flip_array(self, array):
        """flip pixels from 1 to 0 and vice versa"""
        ones = array == 1
        zeros = array == 0

        array[ones] = 0
        array[zeros] = 1

        return array
    
    def clean_small_feats(self, img, size):

        '''
        Removes any patch/feature in a binary image that is below a certian pixel count

        Parameters
        ----------
        img : binary ndarray
        size : (int), minimum patch size needed to be kept in the image

        Returns
        -------
        out : binary ndarray

        '''
        # Label all unique features in binary image
        label, numfeats = ndimage.label(img)

        # Get list of unique feat ids as well as pixel counts
        u, cnt = np.unique(label, return_counts=True)

        # list of feat ids that are too small
        ids = u[cnt < size]

        # Wipe out patches with id that is in `ids` list
        for id in ids:
            label[label==id] = 0

        # Get a list of remaining unique IDs as a check
        u2 = np.unique(label)

        # Convert all labels to 1
        label[label!=0] = 1

        # Feedback
        msg = f'Removing Small Features (<{size}px):'
        print('\n')
        print(f"{msg}\n{'-'*len(msg)}")
        print('Features in: ', len(u)-1) # -1 for 0 (background)
        print('Features out: ', len(u2)-1) # -1 for 0 (background)
        print('Features removed: ', len(ids))

        return label
    
    ## Calcualte morphological charcteristics for each patch in the image
    ## Return df with values, patches, and patch locations to reconstruct image
    def classify_feats(self, img):

        '''
        Calcualtes morphological charcteristics for each patch in the image.

        Returns a df with values, patch area, and patch locations to reconstruct original
        binary image after the target patches are removed

        Input:
        ------
        img : binary ndarray

        Returns:
        --------
        label : (2D array) array where all indv patches have a unique ID (for reference with df)
        df: (dataframe) Contains: PatchID, raw patch area (px), filled patch area (px),
            diameter of circumscribing circle, area of circle, elongation index,
            isolated patch as 2D array, location of each patch within the image

        '''

        # Label individual image features
        label, numfeats = ndimage.label(img)

        # Get list of unique features as well as pixel counts
        ids = np.unique(label)[1:]

        # Find location of every object in label
        locs = ndimage.find_objects(label)

        # Index labeled image for every feature location
        patches = [label[loc] for loc in locs]

        # Isolate just the patch in question for every slice in patches, then convert
        # it to a value of 1, all else to 0
        for i, patch in enumerate(patches):
            patches[i] = np.where(patch==(i+1), 1, 0)

        # Calc raw area for each patch
        p_area = np.array([patch.sum() for patch in patches]) # used later for feedback

        # Calc the circular area for every patch to calc the elongation index
        #  Step 1 - Fill holes in all patches
        #  Step 2 - Erode filled patch, then subtract erosion from filled patch to get boundary
        #  Step 3 - Get image coords for each boundary pixel
        #  Step 4 - Calc distance between each coord, take max value as diameter
        #  Step 5 - Calc circular area for each patch
        #  Step 6 - Divide filled patch area by circle area for index value


        # Step 1 - Fill holes in all patches
        filled = [ndimage.binary_fill_holes(patch).astype(int) for patch in patches]

        ## Calc filled area for each patch
        f_area = np.array([fill.sum() for fill in filled])

        # Step 2 - Erode filled patch, then subtract erosion from filled patch to get boundary
        ero = [ndimage.binary_erosion(fill) for fill in filled]
        bounds = [i - j for i, j in zip(filled, ero)]

        ## Calc boundary count for feedback
        px_count2 = sorted([bound.sum() for bound in bounds[0:5]], reverse=True)

        # Step 3 - Get image coords for each boundary pixel
        coords = [np.transpose(np.nonzero(bound)) for bound in bounds]

        # Step 4 - Calc distance between each coord, take max value as diameter
        di = np.array([spatial.distance.pdist(coord).max() for coord in coords])

        # Step 5 - Calc circular area for each patch
        pr2 = np.pi*(di/2)**2

        #  Step 6 - Divide filled patch area by circle area for index value
        ## I used filled features here because features with holes will be counted
        ## as more elongated without consideration of thier outer boundaries
        ## consider how an empty circle would score
        elong = f_area / pr2

        # Create data frame of all relevant data
        df = pd.DataFrame({'ID': ids,
                        'PatchArea':p_area,
                        'FillArea':f_area,
                        'Diameter': di,
                        'PiR2': pr2,
                        'ElongIndex': elong,
                        'Patch': patches,
                        'PatchLoc': locs
                        }).set_index('ID')

        # Feedback
        msg = f'Classifying Remaining Features (n={len(df.index)})'
        print('\n')
        print(f"{msg}\n{'-'*len(msg)}")
        print('Number of pixels in 5 largest patches: ', np.sort(p_area)[-1:-6:-1])
        print('Number of pixels in 5 largest patches after erosion: ', px_count2)

        return (label, df)
    
    def build_img(self, img, df, th):

        '''
        Rebuild image from df output of `classify_feats()`

        Input:
        ------
        img : (2D array) used solely for image dimensions
        df : dataframe from `classify_feats()`

        Returns:
        --------
        new_img : (2D array) array built from patches that satisfy morph criteria

        '''
        # Create blank image to receive features
        new_img = np.zeros(img.shape)

        # Query df for records with satisfactory elongation index
        new_df = df[df.ElongIndex < th]

        # Add all satisfactory patches to blank image
        for patch, loc in zip(new_df.Patch, new_df.PatchLoc):
            new_img[loc] += patch

        # Feedback
        msg = f'Filtering image for circular patches (ElongIndex > {th})'
        print('\n')
        print(f"{msg}\n{'-'*len(msg)}")
        print('Features in: ', len(df.index))
        print('Features out: ', len(new_df.index))
        print('Features removed: ', len(df.index) - len(new_df.index))

        return new_img


    def master_denoise(self, arr, small_feats_size, elongation_threshold):
        
        # Remove single pixel/very small objects from image
        close_arr = ndimage.binary_closing(arr).astype(int)
        open_arr = ndimage.binary_opening(close_arr).astype(int)

        ## Remove features that are small enough to only be considered noise
        clean = self.clean_small_feats(open_arr, small_feats_size)

        ## Calculate elongation criteria for all features, return info in df
        img, df = self.classify_feats(clean)

        ## Build new image from features with satisfactory elongation values
        new_img = self.build_img(img, df, elongation_threshold)

        return new_img
        
    def execute(self):
        """
        Perform the denoising process on both the foreground and background objects in an image
        
        Workflow:
        1. Efficiently remove very small objects from the image with binary opening and closing
        2. Remove all other image objects with an area smaller than the given threshold
        3. Remove image objects that are not sufficiently elongated (ridges are elongated)
        """

        # Open raster
        raster = rasterio.open(self.raster_path)
        profile = raster.profile
        array  = raster.read(1)

        # Record location of Nans in a mask
        nan_mask = np.isnan(array)

        # Remove errant features from swale areas
        array = self.master_denoise(array, self.small_feats_size, self.elongation_threshold)

        # Flip array 
        array = self.flip_array(array)

        # Remove errant features from ridge areas
        array = self.master_denoise(array, self.small_feats_size, self.elongation_threshold)

        # Flip array back to original classification
        array = self.flip_array(array)

        # Replace Nans
        array[nan_mask] = np.nan

        # Write to disk
        with rasterio.open(self.out_path, "w", **profile) as dst:
            dst.write(array, 1)

        return self.out_path