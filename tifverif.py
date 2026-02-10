import rasterio

# Replace with the actual path to your downloaded file
l8_path = 'Mississippi_L8_6Bands.tif' 

try:
    with rasterio.open(l8_path) as src:
        print("✅ SUCCESS! File is valid.")
        print(f"Count: {src.count} bands")
        print(f"Width: {src.width}, Height: {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Data Type: {src.dtypes[0]}")
        
except Exception as e:
    print("❌ ERROR: File is corrupted.")
    print(e)