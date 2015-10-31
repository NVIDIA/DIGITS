from urlparse import urlparse, parse_qs

def get_workspace(url):
	try:
		workspace_name = parse_qs(urlparse(str(url)).query, keep_blank_values=True)["workspace"][0].encode('utf-8')
		if workspace_name == '':
			workspace_name = "Default"
	except:
		workspace_name = "Default"
	return workspace_name