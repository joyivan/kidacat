import pydicom
baseDir='/Users/lizirong/Downloads/CT_MD/COVID-19_Cases'
b=[]
for i in range(153):
	a=pydicom.dcmread(baseDir+'/P'+str(i+1).zfill(3)+'/IM0001.dcm')
	b.append(a.SliceThickness)
print(b)

	

