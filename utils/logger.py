import datetime
import sys
import os

class Logger(object):
	def __init__(self, log_fn=None, prefix=""):
		"log filename"
		if os.path.exists(log_fn):
			os.remove(log_fn)
		self.log_fn = log_fn

		self.prefix="" 
		if prefix:
			self.prefix = prefix + ' | '
		
		self.file_pointers = []

	def add_line(self, content):
		"""
		add content to file or print content directly

		Args:
		    content ([type]): [description]
		"""
		msg = self.prefix + content
		fp = open(self.log_fn, 'a')
		fp.write(msg + '\n')
		fp.flush()
		fp.close()
		print(msg)
		sys.stdout.flush()
			
def print_dict(logger, d, ident=''):
	for k in d:
		if isinstance(d[k], dict):
			logger.add_line("{}{}".format(ident, k))
			print_dict(d[k], ident='  '+ident)
		else:
			logger.add_line("{}{}: {}".format(ident, k, str(d[k])))

