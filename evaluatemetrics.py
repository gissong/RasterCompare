#author: Song Gao Email:song.gao@wisc.edu
import sys
import numpy as np
import gdal
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

# get Raster format driver (e.g., GTiff, PNG, JPEG, ASCII Grid) 
# more driver information from GDAL website: https://www.gdal.org/formats_list.html
fmt = 'GTiff'
driver = gdal.GetDriverByName(fmt)

# open input dataset
ds1 = gdal.Open('input1.tif')
ds2 = gdal.Open('input2.tif')
if ds1 == None or ds2 == None:
	print ("can't open file")
	sys.exit()
print ('ds1 number of bands: '+str(ds1.RasterCount))
print ('ds2 number of bands: '+str(ds2.RasterCount))

# create output dataset
cols = ds1.RasterXSize
rows = ds1.RasterYSize
print ('ds1 number of cols: '+str(cols) + ' number of rows: '+str(rows) )
cols = ds2.RasterXSize
rows = ds2.RasterYSize
print ('ds2 number of cols: '+str(cols) + ' number of rows: '+str(rows) )

# calculate output array
array_true = ds1.GetRasterBand(1).ReadAsArray().astype(np.float)
array_pred = ds2.GetRasterBand(1).ReadAsArray().astype(np.float)

#Convert NumPy array to binary vector based on a threshold (optional, this is required for Binary classification result only)
array_pred = np.where(array_pred >= 1, 1, 0)
array_true = np.where(array_true >= 1, 1, 0)

#Output the difference image between two input rasters
outputArr = array_true - array_pred
dsOut = driver.Create('diff.tif', cols, rows, 1, gdal.GDT_Float32)
#write band to the diff raster
band = dsOut.GetRasterBand(1)
band.SetColorInterpretation(gdal.GCI_GrayIndex)
band.WriteArray(outputArr)
dsOut = None

# close the input raster files
ds1 = None
ds2 = None

#Evaluation metrics (i.e., precision, recall, F1-score)
eval_f1_score = f1_score(array_true, array_pred, average='micro')
print ("f1_score: "+str(eval_f1_score))
precision_recall_fscore = precision_recall_fscore_support(array_true, array_pred, average='weighted')
print ("precision_recall_fscore_support: "+str(precision_recall_fscore))
