from utils import *

vhosts = get_top_vhosts(top_n=15)
guids = get_guids_by_vhost(*vhosts, least=235)
#labels = get_labels(*guids, vendor=True)
download_and_convert(*guids, n_jobs=8)
