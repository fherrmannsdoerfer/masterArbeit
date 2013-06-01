import numpy as np
import vigra
import math
from scipy import stats
import matplotlib.pyplot as plot
import random

def smoothGaussian(array,width=5):
    width_filter = 2*width + 1
    filter_gauss = np.ones([width_filter])
    sigma = np.sqrt(width)
    if width_filter > len(array):
        print "array is smaller than filter width (filterwidth = 2*width+1)"
    
    for i in range(width_filter):
        filter_gauss[i] = 1./sigma/np.sqrt(2*np.pi)*np.exp(-.5*((i-(width_filter/2))/(sigma))**2)
    
    add_at_begining = np.ones(width_filter/2) * array[0]
    add_at_end      = np.ones(width_filter/2) * array[-1]
    
    extended_array = np.hstack([add_at_begining, array, add_at_end])
    #smoothed_array = np.zeros(len(array))
    smoothed_array2 = np.zeros(len(array))
    #for i in range(len(array)):
    #    for j in range(width_filter):
    #        smoothed_array[i] = smoothed_array[i] + extended_array[i+j] * filter_gauss[j]
    
    for i in range(width,int(len(array)- width)):
        smoothed_array2[i] = np.sum(array[i-width_filter/2:i+width_filter/2+1] * filter_gauss)

    return smoothed_array2 

def smoothGaussianZ(array,width=5):
    width_array = array.shape[0]
    height_array = array.shape[1]
    width_filter = 3*width
    filter_gauss = np.ones([width_filter])
    sigma = np.sqrt(width)
    if width_filter > array.shape[2]:
        print "array is smaller than filter width (filterwidth = 3*width)"
    
    for i in range(width_filter):
        filter_gauss[i] = 1./sigma/np.sqrt(2*np.pi)*np.exp(-.5*((i-(width_filter/2))/(sigma))**2)
    
    add_at_begining = np.zeros([width_array, height_array,width_filter/2])
    #add_at_begining[...,0:-1] = array[...,0]
    add_at_end      = np.zeros([width_array, height_array,width_filter/2])
    #add_at_end[...,0:-1] = array[...,-1]
    
    extended_array = np.concatenate([add_at_begining, array, add_at_end], axis = 2)
    smoothed_array = np.zeros_like(array)
    for i in range(array.shape[2]):
        for j in range(width_filter):
            smoothed_array[...,i] = smoothed_array[...,i] + extended_array[...,i+j] * filter_gauss[j]
        print i
    return smoothed_array 


def find_bead_positions(frame):
    [width, height] = frame.shape[:2]
    
    limit = np.mean(frame)*1.3
    posX=[]
    posY=[]
    frame[0:10,0:height]=0
    frame[width-10:width,0:height]=0
    frame[0:width,0:10]=0
    frame[0:width,height-10:height]=0
    while 1==1:
        [tempposX,tempposY]=np.where(frame==np.max(frame))
        if frame[tempposX[0], tempposY[0]]<limit:
            break
        else:
            posX.append(tempposX[0])
            posY.append(tempposY[0])
            frame[tempposX[0]-10:tempposX[0]+10,tempposY[0]-10:tempposY[0]+10] = 0
    
    return posX, posY

def getDataArray(filename, width, height, stacksize):
    try:    # try to load cached numpy file format (faster)
        frames = np.load(filename.split('.')[0:-1][0]+'.npy')
        print "I read the cached file."
        return frames
    except IOError: # file could not be read
        pass
    fpa = 50
    frames = np.zeros([width, height, stacksize])
    for i in range(int(stacksize/fpa)):
        frames[...,i*fpa:(i+1)*fpa] = vigra.impex.readSifBlock(filename,i*fpa,(i+1)*fpa)
        print 'progress: %03i percent' %int(i*fpa/stacksize *100.)
 
    a=int(stacksize/fpa) * fpa
    #print a
    for i in range(int(stacksize%fpa)):
        frames[...,a+i] = vigra.impex.readSifBlock(filename,a+i,a+i+1)[...,0]
    print 'progress: %03i percent' %int(100)
    np.save(filename.split('.')[:-1][0], frames)
    return frames
    
def find_background_pixel(posXBeads, posYBeads, frames):
    dimx = frames.shape[0]
    dimy = frames.shape[1]
    while 1==1:
        posXBackground = np.random.randint(0, dimx)
        posYBackground = np.random.randint(0, dimy)
        if posXBackground in posXBeads and posYBackground in posYBeads:
            continue
        else:
            break
        
    return posXBackground, posYBackground

def find_number_bead_and_background(number_beads, number_background, frames, filename):
    if number_background == 0:
        print "number of backgroundpixel must be larger 0"
        return 0
    
    posXBeads, posYBeads = find_bead_positions(filename)
    
    posXBackground = []
    posYBackground = []
    for i in range(number_background):
        tmpX, tmpY = find_background_pixel(posXBeads, posYBeads, frames[...,0])
        posXBackground = np.hstack([posXBackground, tmpX])
        posYBackground = np.hstack([posYBackground, tmpY])

    if posXBeads:
        indices = np.random.randint(0,len(posXBeads), number_beads)  
        positions_to_check = np.vstack([np.hstack([posXBackground, posXBeads[indices]]),np.hstack([posYBackground, posYBeads[indices]])])    
        #print positions_to_check
    else:
        positions_to_check = np.vstack([posXBackground, posYBackground])
        #print positions_to_check
        
    return positions_to_check

def find_good_candidate(number_points,frame0,skip = 0):
    frame0 = np.mean(frame0,2)
    maxIntensity = np.max(frame0)
    minIntensity = np.min(frame0)
    step_size = (maxIntensity - minIntensity)/float(number_points)
    allrang = 1.0/100 #allowed_range_in_percent_of_step_size
    
    positions_to_check = []
    for i in range(number_points):
        posXlower, posYlower = np.where(i*step_size + minIntensity - allrang*step_size < frame0 )
        posX = []
        posY = []
        for j in range(len(posXlower)):
            if frame0[posXlower[j], posYlower[j]] < (i+1)*step_size + minIntensity + allrang*step_size:
                posX.append([posXlower[j]])
                posY.append([posYlower[j]])
        if posX:
            index = np.random.randint(0,len(posX))
            positions_to_check.append([posX[index][0], posY[index][0]])            
        else:
            if skip == 0:
                #print "no point in range used a random point as %ith point instead" %i
                positions_to_check.append([np.random.randint(0,frame0.shape[0]),np.random.randint(0,frame0.shape[1])])
            else:
                print "no point in range skipped %ith point instead" %i
    
    return np.transpose(np.array(positions_to_check))
    
def print_number_on_image(image_org, number, position, scale=1):
    image = np.copy(image_org)
    number = int(number)
    size_bar = 4 * scale
    space_between_numbers = 3
    shiftX = (size_bar+space_between_numbers)
    n0=[1,1,1,0,1,1,1]
    n1=[0,0,1,0,0,1,0]
    n2=[1,0,1,1,1,0,1]
    n3=[1,0,1,1,0,1,1]
    n4=[0,1,1,1,0,1,0]
    n5=[1,1,0,1,0,1,1]
    n6=[0,1,0,1,1,1,1]
    n7=[1,0,1,0,0,1,0]
    n8=[1,1,1,1,1,1,1]
    n9=[1,1,1,1,0,1,1]
    
    max_val = np.max(image)
    
    digits = []
    number_copy = np.copy(number)
    number_digits = int(np.ceil(np.log10(number+1)))
    
    for i in range(number_digits):
        digits.append([np.mod(number_copy,10)])
        number_copy = number_copy / 10
    
    
    for i in range(number_digits):
        
        actual_digit = digits[number_digits-i-1][0]
        number = number / 10
        
        if actual_digit == 0:
            segments = n0
        elif actual_digit == 1:
            segments = n1
        elif actual_digit == 2:
            segments = n2
        elif actual_digit == 3:
            segments = n3
        elif actual_digit == 4:
            segments = n4
        elif actual_digit == 5:
            segments = n5
        elif actual_digit == 6:
            segments = n6
        elif actual_digit == 7:
            segments = n7
        elif actual_digit == 8:
            segments = n8
        elif actual_digit == 9:
            segments = n9
        
        if position[1] +(number%10+1)*(size_bar+space_between_numbers)<image.shape[1] and position[0]+2*size_bar <image.shape[0]:
            if segments[0]:
                image[position[0],i*shiftX + position[1]:i*shiftX + position[1]+size_bar+1]=max_val
            if segments[1]:
                image[position[0]:position[0]+size_bar,i*shiftX + position[1]]=max_val
            if segments[2]:
                image[position[0]:position[0]+size_bar,i*shiftX + position[1]+size_bar]=max_val
            if segments[3]:
                image[position[0]+size_bar-0,i*shiftX + position[1]:i*shiftX + position[1]+size_bar+1]=max_val
            if segments[4]:
                image[position[0]+size_bar:position[0]+2*size_bar,i*shiftX + position[1]]=max_val
            if segments[5]:
                image[position[0]+size_bar:position[0]+2*size_bar+1,i*shiftX + position[1]+size_bar]=max_val
            if segments[6]:
                image[position[0]+2*size_bar-0,i*shiftX + position[1]:i*shiftX + position[1]+size_bar+1]=max_val
        else:
            print 'number does not fit in at the given position:',position
        
    return image

def skellam_paramterer(frames, x, y, filter_width):
    width, height, stacksize = frames.shape
    diffMat=np.zeros([stacksize-1])
    muSMat=0
    sigma2Mat = 0
    mu1 = 0
    mu2 = 0
    for i in range(int(stacksize)-1):
        diffMat[i]=frames[x,y,i]-frames[x,y,i+1]
    muSMat = np.sum(diffMat,0)/stacksize
    #muSMat = np.mean(frames, 2)
   
    for i in range(int(stacksize)-1):
        sigma2Mat += (muSMat-(diffMat[i]))**2
    sigma2Mat = sigma2Mat / (stacksize-1)
    mu1 = (muSMat+sigma2Mat)/2.
    mu2 = (-muSMat+sigma2Mat)/2.

    return mu1, mu2

def skellam_paramterers(frames, x, y, filter_width):
    if x>frames.shape[0] or y > frames.shape[1]:
        print 'x or y is too big'
    width, height, stacksize = frames.shape
    diffMat = np.zeros([stacksize-1])
    sigma2 = np.zeros([stacksize-2*filter_width])
    muS = np.zeros([stacksize-2*filter_width])
    #framesS = np.where(frames[x,y,:] < 1.2*np.mean(frames[x,y,:]),frames[x,y,:], np.mean(frames[x,y,:]))   
    diffMat = frames[x,y,:-1] - frames[x,y,1:]
    #diffMat = framesS[:-1] - framesS[1:]
        
    for k in range(filter_width, stacksize-filter_width):
        muS[k-filter_width] = np.sum(diffMat[k-filter_width: k+filter_width])/(2*filter_width)
        sigma2[k-filter_width] = np.sum((muS[k-filter_width] - diffMat[k-filter_width:k+filter_width])**2) / (2*filter_width-1)
    mu1 = (muS+sigma2)/2.
    mu2 = (-muS+sigma2)/2.

    return mu1, mu2

def calc_skellam_paramterers(frames, x, y):
    #this function calculates the skellam parameters at position x,y
    if x>frames.shape[1] or y > frames.shape[0]:
        print 'x or y is too big'
    width, height, stacksize = frames.shape
    diffMat = np.zeros([stacksize-1])
    sigma2 = 0
    muS = 0
    
    for i in range(int(stacksize-1)):
        diffMat[i] = frames[x,y,i]-frames[x,y,i+1]
    
    muS = np.sum(diffMat)/stacksize
    sigma2 = np.var(diffMat)

    mu1 = (muS+sigma2)/2.
    mu2 = (-muS+sigma2)/2.

    return mu1, mu2

def get_random_points(width, height, number_points):
    points = np.zeros([number_points, 2])
    for i in range(number_points):
        points[i,0] = np.random.randint(0, width-1)
        points[i,1]  =np.random.randint(0, height-1)
    
    return points

def nanmean(data):
    return np.mean(data[np.isfinite(data)])

def nanvar(data):
    return np.var(data[np.isfinite(data)])

def poisson(x, lamb):
    #returns poisson distribution, maximal lamb value = 100 for higher values zeros are returned
    y = np.zeros(len(x))
    if lamb > 50:
        lamb = int(lamb)
    if lamb > 100:
        return y
    if len(x) <= 1:
        y[0] = float(pow(lamb,x[0])) / float(math.factorial(x[0])) * float(np.exp(-lamb))
    if len(x) <= 2:
        y[1] = float(pow(lamb,x[1])) / float(math.factorial(x[1])) * float(np.exp(-lamb))
    if len(x) > 2:
        for i in range(2,len(x)):
            if i > lamb and y[i-1] < 1e-25:
                return y
            else:
                a = math.log(pow(lamb,x[i]))
                b = math.log(math.factorial(x[i]))
                c = np.exp(-lamb)
                d = a - b
                e = np.exp(d)
                y[i] = e * c
    return y  

def calc_a_b_with_scellam(frames, num_points = 10, filter_width = 195, list_param = [10,15,50,15]):


    [width, height, stacksize] = frames.shape
    
    #points = mT.get_random_points(width, height, 10)
    points = find_good_candidate(num_points, frames)
    points = np.transpose(points)
    #x_val = np.zeros([points.shape[0]])
    list_x_val = []
    list_mu1 = []
    list_mu2 = []
    col = []
    import matplotlib.pyplot as plot
    for j in range(points.shape[0]):
        #print '(%i/%i)'%(j,points.shape[0])
        mu1, mu2 = skellam_paramterers(frames, points[j,0], points[j,1], filter_width)
        list_mu1 = np.hstack([list_mu1, mu1])
        list_mu2 = np.hstack([list_mu2, mu2])
        col.append(np.ones(len(mu1))*j)
        #mu1, mu2 = mT.skellam_paramterer(frames, points[j,0], points[j,1], filter_width)
        x_val = smoothGaussian(frames[points[j,0], points[j,1],:], filter_width)
        #rand wird nicht mitbenutzt
        list_x_val = np.hstack([list_x_val, x_val[filter_width:-filter_width]])
        #plot.figure(j)
        #plot.scatter(list_x_val, list_mu1, c = col)
        #plot.figure(j+points.shape[0])
        #plot.plot(range(len(frames[points[j,0],points[j,1],:])),frames[points[j,0],points[j,1],:] )
        #x_val = np.mean(frames[points[j,0], points[j,1],:])
        
        #plot.scatter(x_val, np.mean(mu1))
    
    #plot.figure(1)
    #plot.scatter(list_x_val, list_mu1, c = col)
    #print points
   
    m1,c1,_,_,_ = stats.linregress(list_x_val, list_mu1)
    m2,c2,_,_,_ = stats.linregress(list_x_val, list_mu2)
    x = range(int(np.min(list_x_val)), int(np.max(list_x_val)))
    y = m1 * np.array(x) + c1
    #plot.plot(x,y)
    np_x_val = np.array(list_x_val)

    #plot.xlabel('mean (smoothed)')
    #plot.ylabel('scellam parameter')
    #plot.title('before RANSAC')
    sort_index = np.argsort(list_x_val)
    
    data = np.vstack([list_x_val[sort_index],list_mu1[sort_index]])
    
    
    mmin,cmin, idx = take_lowest_points_for_calculation(np.vstack([data[0,:], data[1,:]]).T, 2000, list_param[3])
    
    mran, cran,idx = do_linear_ransac(np.vstack([data[0,:], data[1,:]]).T, 2000, list_param[0])
    
    
    
    
    mrw, crw = do_linear_ransac_with_weighting(np.vstack([data[0,:], data[1,:]]).T, 2000, list_param[1],list_param[2])
    fig = plot.figure(1)
    plot.subplot(1,3,2)
    plot.scatter(list_x_val, list_mu1, c = col)
    y = mrw * np.array(x) + crw
    plot.plot(x,y)
    plot.xlabel('mean (smoothed)')
    plot.ylabel('scellam parameter')
    plot.title('after RANSAC with weights')
    
    
    #mran, cran,idx = do_linear_ransac(np.vstack([list_x_val, list_mu1]).T, 500)
    
    plot.subplot(1,3,1)
    plot.scatter(list_x_val, list_mu1, c = col)
    y = mran * np.array(x) + cran
    plot.plot(x,y)
    plot.xlabel('mean (smoothed)')
    plot.ylabel('scellam parameter')
    plot.title('after RANSAC')
    #plot.plot(data[0,idx], data[1,idx], 'bo', markersize = 10)
    
    plot.subplot(1,3,3)
    plot.scatter(list_x_val, list_mu1, c = col)
    y = mmin * np.array(x) + cmin
    plot.plot(x,y)
    plot.xlabel('mean (smoothed)')
    plot.ylabel('scellam parameter')
    plot.title('after just minimal values fit')
    
    #print m1,mran, mrw, mmin
    #print -c1/m1,-cran/mran, -crw/mrw, -cmin/mmin
    #plot.show()
    return mran ,-cran/mran, mrw,  -crw/mrw, mmin, -cmin/mmin, fig

def do_linear_ransac(points, nbr_loops, percent = 10):
    
    n = points.shape[0]
    points = points[0:int(n/100.*percent),:]
    anz = 4
    m = []
    c = []
    err = []
    indx = []
    
    for k in range(nbr_loops):
        #print k
        #np.random.shuffle(points)
        indices = random.sample(range(points.shape[0]), anz)
        m1,c1,_,_,error = stats.linregress(points[indices,0], points[indices,1])
        #plot.scatter(points[:anz,0], points[:anz,1])
        '''plot.plot(points[indices,0], points[indices,1], 'bo', markersize = 10)
        x = range(int(np.min(points[indices,0])),int(np.max(points[indices,0])))
        y = m1*np.array(x) + c1
        plot.plot(x,y)
        plot.scatter(points[:,0],points[:,1])
        #print error
        error = np.sum((points[:,1]-(c1+m1*points[:,0]))**2)
        print error
        plot.show()'''
        indx.append(indices)
        m.append(m1)
        c.append(c1)
        
        error = np.sum((points[:,1]-(c1+m1*points[:,0]))**2)
        
        err.append(error)
    
    idx = np.where(err == np.min(err))[0][0]
    return m[idx], c[idx], indx[idx]
    

def lin_fun(x,m,c):
    return m*x+c 

def do_linear_ransac_with_weighting(allpoints, nbr_loops, anz_intervals = 15, percent= 50):
    #plot.show()
    from scipy.optimize import curve_fit as cfit
    n = allpoints.shape[0]
    anz = int(n/100. * percent)
    #anz_intervals = int(anz / 10)
    borders = range(int(np.min(allpoints[:,0])),int(np.max(allpoints[:,0])), int(np.ceil((np.max(allpoints[:,0])-np.min(allpoints[:,0]))/anz_intervals)))
    borders.append(np.max(allpoints[:,0]))
    borders = np.array(borders)
    variances = []
    
    m = []
    c = []
    err = []
    indx = []
    
    import random
    for k in range(nbr_loops):
        #print k
        variances = []
        indices = random.sample(range(n), anz)
        points = allpoints[indices,:]
        sort_index = np.argsort(points[:,0])
    
        points = points[sort_index,:]
        
        for i in range(len(borders)-1):
            pos0 = np.where(points[:,0]<=borders[i+1])
            index = pos0[0][np.where(points[pos0,0]>borders[i])[1]]
            for j in range(len(index)):
                if len(index) < 5:
                    variances.append(99999999)  #bei nur einem Punkt wird dieser kaum gewichtet
                    #print i
                else:
                    variances.append(np.var(points[index,1]))
     
        variances = np.array(variances) / np.min(variances)
        sigma = variances
        
        [mw,cw],_ = cfit(lin_fun, points[:,0], points[:,1], sigma = sigma)

        
        x = range(int(np.min(points[:,0])),int(np.max(points[:,0])))
        y = mw*np.array(x) + cw
        
        '''for i in range(len(borders)):
            plot.plot([borders[i],borders[i]], [np.min(points[:,1]),np.max(points[:,1])])
        plot.plot(x,y)
        plot.scatter(points[:,0],points[:,1], c = sigma)'''
        #print error
        #error = np.sum((points[:,1]-(cw+mw*points[:,0]))**2)
        #print error
        #plot.show()
        indx.append(indices)
        m.append(mw)
        c.append(cw)
        
        #error = np.sum((allpoints[:,1]-(cw+mw*allpoints[:,0]))**2)
        
        #err.append(error)
    
    
    
    return np.median(m), np.median(c)

def take_lowest_points_for_calculation(allpoints, nbr_loops, anz_intervals = 15):
    
    n = allpoints.shape[0]
    #anz_intervals = int(n / 10)
    borders = range(int(np.min(allpoints[:,0])),int(np.max(allpoints[:,0])), int(np.ceil((np.max(allpoints[:,0])-np.min(allpoints[:,0]))/anz_intervals)))
    borders.append(np.max(allpoints[:,0]))
    
    list_x = []
    list_y = []
    
    for i in range(len(borders)-1):
        pos0 = np.where(allpoints[:,0]<=borders[i+1])
        index = pos0[0][np.where(allpoints[pos0,0]>borders[i])[1]]
        try:
            idx_min_mu1 = index[np.where(allpoints[index,1] == np.min(allpoints[index,1]))[0][0]]
            list_x.append(allpoints[idx_min_mu1,0])
            list_y.append(allpoints[idx_min_mu1,1])
        except:
            ' '

    anz = 2
    m = []
    c = []
    err = []
    indx = []
    list_x = np.array(list_x)
    list_y = np.array(list_y)
    for k in range(nbr_loops):
        #print k
        #np.random.shuffle(points)
        indices = random.sample(range(len(list_x)), anz)
        m1,c1,_,_,error = stats.linregress(list_x[indices],list_y[indices])
        
        error = np.sum(np.abs(list_y-(c1+m1*list_x))) #nicht die summe der Quadrate um die Aussreiser nicht so stark zu gewichten
        
        '''plot.plot(list_x[indices],list_y[indices], 'bo', markersize = 10)

        x = range(int(np.min(allpoints[:,0])),int(np.max(allpoints[:,0])))
        y = m1*np.array(x) + c1
        plot.plot(x,y)
        #plot.scatter(allpoints[:,0],allpoints[:,1])
        plot.scatter(list_x,list_y)
        #print error
        
        
        print error
        plot.show()'''
       
        indx.append(indices)
        m.append(m1)
        c.append(c1)
        
        error = np.sum(np.abs(list_y-(c1+m1*list_x)))        
        err.append(error)
    
    idx = np.where(err == np.min(err))[0][0]    #nachdem hoffentlich eine gute Steigung gefunden ist sollen alle punkte die in der Naehe dieser Geraden liegen mit einbezogen werden 
    points_to_fit_x = []
    points_to_fit_y = []
    for i in range(n):
        if (allpoints[i,1]-(c[idx] + m[idx]*allpoints[i,0]))**2 < 80**2:
            points_to_fit_x.append(allpoints[i,0])
            points_to_fit_y.append(allpoints[i,1])
    #plot.plot(points_to_fit_x,points_to_fit_y, 'bo', markersize = 10)
    #plot.scatter(allpoints[:,0],allpoints[:,1])
    mfin,cfin,_,_,error = stats.linregress(points_to_fit_x,points_to_fit_y)
    
    return m[idx], c[idx], indx[idx]
    
    