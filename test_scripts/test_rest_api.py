"""
Author : Mohit Jain
Email  : develop13mohit@gmail.com

USAGE --

Call this script after having the digits server kept running at (localhost:5000).
"""

import subprocess
import os
import os.path

def test_classify_one(job_id, image_file):
    '''
    Test the classify_one method.
    '''
    curl_req = 'curl localhost:5000/models/images/classification/classify_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if 'predictions' in resp_parsed:
            return True
        else:
            return False

def test_visualize_one(job_id, image_file):
    '''
    Test the classify_one method with the visualize_one.json request
    '''
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y'
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if 'visualizations' in resp_parsed:
            return True
        else:
            return False

def test_visualize_one_specific_layer(job_id, image_file):
    '''
    Test the classify_one method with the visualize_one.json request and only get a specific layer as output.
    '''
    layer = 'conv1'
    not_layer = 'fc6'
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F select_visualization_layer='+layer
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if layer in resp_parsed and not_layer not in resp_parsed:
            return True
        else:
            return False

def test_visualize_one_incorrect_layer(job_id, image_file):
    '''
    Test the classify_one method with the visualize_one.json request and get error output.
    '''
    layer = 'mohit'
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F select_visualization_layer='+layer
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        error = '404'
        if layer not in resp_parsed and error in resp_parsed:
            return True
        else:
            return False

def test_visualize_one_save_mat(job_id, image_file,save_file_location):
    '''
    Test the classify_one method with the visualize_one.json request and save as .mat file
    '''
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F save_vis_file=y -F save_type_mat=y -F save_vis_file_location='+save_file_location
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        if os.path.isfile(save_file_location+'/visualization_'+job_id+'.mat'):
            os.system('rm '+save_file_location+'/visualization_'+job_id+'.mat')
            return True
        else:
            return False

def test_visualize_one_save_numpy(job_id, image_file,save_file_location):
    '''
    Test the classify_one method with the visualize_one.json request and save as .npy file
    '''
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F save_vis_file=y -F save_type_numpy=y -F save_vis_file_location='+save_file_location
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        if os.path.isfile(save_file_location+'/visualization_'+job_id+'.npy'):
            os.system('rm '+save_file_location+'/visualization_'+job_id+'.npy')
            return True
        else:
            return False

def test_visualize_one_specific_layer_save_mat(job_id, image_file,save_file_location):
    '''
    Test the classify_one method with the visualize_one.json request and save as .mat file
    '''
    layer = 'conv1'
    not_layer = 'fc6'
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F save_vis_file=y -F save_type_mat=y -F save_vis_file_location='+save_file_location
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        if os.path.isfile(save_file_location+'/visualization_'+job_id+'.mat'):
            os.system('rm '+save_file_location+'/visualization_'+job_id+'.mat')
            return True
        else:
            return False
"""
   else:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if layer in resp_parsed and not_layer not in resp_parsed and os.path.isfile(save_file_location+'/visualization_'+job_id+'.mat'):
            os.system('rm '+save_file_location+'/visualization_'+job_id+'.mat')
            return True
        else:
            return False
"""

def test_visualize_one_incorrect_specific_layer_save_mat(job_id, image_file,save_file_location):
    '''
    Test the classify_one method with the visualize_one.json request and save as .mat file
    '''
    layer = 'mohit'
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F save_vis_file=y -F save_type_mat=y -F save_vis_file_location='+save_file_location
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        error = '400'
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if error in resp_parsed:
            return True
        else:
            return False


def test_visualize_one_specific_layer_save_numpy(job_id, image_file,save_file_location):
    '''
    Test the classify_one method with the visualize_one.json request and save as .npy file
    '''
    layer = 'conv1'
    not_layer = 'fc6'
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F save_vis_file=y -F save_type_numpy=y -F save_vis_file_location='+save_file_location
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        if os.path.isfile(save_file_location+'/visualization_'+job_id+'.npy'):
            os.system('rm '+save_file_location+'/visualization_'+job_id+'.npy')
            return True
        else:
            return False

def test_visualize_one_incorrect_specific_layer_save_numpy(job_id, image_file,save_file_location):
    '''
    Test the classify_one method with the visualize_one.json request and save as .numpy file to return error.
    '''
    layer = 'mohit'
    curl_req = 'curl localhost:5000/models/images/classification/visualize_one.json -XPOST -F job_id='+job_id+' -F image_file=@'+image_file+' -F show_visualizations=y -F save_vis_file=y -F save_type_numpy=y -F save_vis_file_location='+save_file_location
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    if err:
        return False
    else:
        error = '400'
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if error in resp_parsed:
            return True
        else:
            return False


"""
    else:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace(",","").split()
        if layer in resp_parsed and not_layer not in resp_parsed and os.path.isfile(save_file_location+'/visualization_'+job_id+'.npy'):
            os.system('rm '+save_file_location+'/visualization_'+job_id+'.npy')
            return True
        else:
            return False
"""

def test_API(digits_up, job_id, image_file, save_file_location):
    '''
    The main function calling all methods for testing each element.
    '''
    curl_req = 'curl localhost:5000/index.json'
    print curl_req
    req = subprocess.Popen([curl_req], stdout=subprocess.PIPE, shell=True)
    (resp, err) = req.communicate()
    try:
        resp_parsed = resp.replace("\"","").replace("\n","").replace("{","").replace("}","").replace(":","").replace("'","").split()
        if resp_parsed[0] == 'models' or resp_parsed[0] == 'datasets':
            digits_up = 'Yes'
    except:
        digits_up = 'No'

    if digits_up == 'No':
        print "\nError : DIGITS server is not running. Aborting Tests.\n############################################################################\n"
        return
    log_str = """
############################################################################
[CHECK] : DIGITS server up --- %s

Testing API request now ...
        """ 
    print log_str % digits_up

    test_ctr = 0
    test_status = 'FAIL'
    total_test_ctr = 10


    # Run the tests from here on.
    if test_classify_one(job_id, image_file):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing classify one image --- %s\n" % test_status
    test_status = 'FAIL'

    if test_visualize_one(job_id, image_file):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image --- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_specific_layer(job_id, image_file):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image with specific layer --- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_save_mat(job_id, image_file, save_file_location):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image with saving .mat file of visualization --- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_save_numpy(job_id, image_file, save_file_location):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image with saving .npy file of visualization --- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_specific_layer_save_mat(job_id, image_file, save_file_location):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image for a specific layer with saving .mat file of visualization --- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_specific_layer_save_numpy(job_id, image_file, save_file_location):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image for a specific layer with saving .npy file of visualization --- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_incorrect_layer(job_id, image_file):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image for a specific layer (which is not in network)--- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_incorrect_specific_layer_save_mat(job_id, image_file, save_file_location):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image for a specific layer (which is not in network) and saving its .mat--- %s\n" % test_status
    test_status = 'FAIL'
    
    if test_visualize_one_incorrect_specific_layer_save_numpy(job_id, image_file, save_file_location):
        test_status = 'PASS'
        test_ctr+=1
    print "\n[Check] : Testing visualize one image for a specific layer (which is not in network) and saving its .mat--- %s\n" % test_status
    test_status = 'FAIL'
    
    
    log_str = """

############################################################################
Testing Complete:

[RESULT] : %s of %s tests completed successfully. [%s percent]

############################################################################
              """
    print log_str % (str(test_ctr),str(total_test_ctr),str((test_ctr*100)/total_test_ctr))
    return

if __name__=='__main__':
    
    glob_job_id = '20150625-051205-c433'
    glob_image_file = '/home/mohit/Downloads/cat.jpg'
    glob_save_file_location = '/home/mohit/Downloads'
    
    # Check if DIGITS is running.
    init_str = """
############################################################################
#                      Welcome to REST API Test Suite                      #
############################################################################

Test Job-id : %s
Test Image  : %s

############################################################################
Querying the DIGITS Server :
               """
    print init_str % (glob_job_id, glob_image_file)

    digits_up = 'No'
    test_API(digits_up, glob_job_id, glob_image_file, glob_save_file_location)
