from urlparse import urlparse, parse_qs

def get_workspace(url):
	workspace_name = parse_qs(urlparse(str(url)).query, keep_blank_values=True)["workspace"][0].encode('utf-8')
	return workspace_name

