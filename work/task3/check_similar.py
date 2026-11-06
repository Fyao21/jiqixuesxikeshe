from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.fft import dctn,fft2,ifft2
from scipy.ndimage import gaussian_filter

HERE=Path(__file__).resolve().parent
d=np.load(HERE/'data.npz')
hashes=[]
filtered=[]
for im in d['images']:
    g=np.asarray(Image.fromarray(im).convert('L').resize((32,32)),dtype=float)
    c=dctn(g,norm='ortho')[:8,:8].ravel()[1:]
    hashes.append(c>np.median(c))
    green=np.asarray(Image.fromarray(im[:,:,1]).resize((192,192)),dtype=float)
    high=green-gaussian_filter(green,4)
    yy,xx=np.mgrid[:192,:192]
    high*=((xx-96)**2+(yy-96)**2<85**2)
    high-=high.mean()
    filtered.append(high)
pairs=[]
for i in range(80):
    for j in range(i+1,80):
        h=int(np.count_nonzero(hashes[i]!=hashes[j]))
        if h<=12 or i in [0,1,5] and j in [0,1,5]:
            a,b=filtered[i],filtered[j]
            cross=ifft2(fft2(a)*np.conj(fft2(b))).real/(np.linalg.norm(a)*np.linalg.norm(b))
            window=np.r_[np.arange(11),np.arange(192-10,192)]
            score=float(cross[np.ix_(window,window)].max())
            if score>.4 or i in [0,1,5] and j in [0,1,5]:
                pairs.append({'a':int(d['ids'][i]),'b':int(d['ids'][j]),'phash_distance':h,'vessel_highpass_max_ncc':score})
pairs=sorted(pairs,key=lambda p:-p['vessel_highpass_max_ncc'])
(HERE/'similar_image_audit.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
print(json.dumps(pairs,indent=2))
