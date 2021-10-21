import web_utils
from pprint import pprint
import nltk
import re
import yaml
from pathlib import Path
import csv
import argparse

keywords = [
    'fair',
    'bias',
    'discriminate',
    'aware',
    'balance',
    'disparity',
    'opportunity',
    'audit',
    'gender',
    'stereotype',
    'race',
    'amplification'
]

links = {
    'NIPS': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/nips/neurips2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/nips/nips2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/nips/nips2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/nips/nips2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/nips/nips2016.bht%3A&h=1000&format=json'
    ],
    'ICML': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icml/icml2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icml/icml2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icml/icml2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icml/icml2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icml/icml2016.bht%3A&h=1000&format=json'
    ],
    'FSE': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2016.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2015.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2014.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2013.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2012.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2011.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2010.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2009.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2008.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2007.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2006.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2005.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2004.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2003.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2002.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2001.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse2000.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/esec/esec99.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse98.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/esec/esec97.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse96.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/esec/esec95.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/sigsoft/fse94.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/esec/esec93.bht%3A&h=1000&format=json'
    ],
    'ICSE': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2016.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2014.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2013.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2012.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2011.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2009.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2008.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2007.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2006.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2005.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2004.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2003.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2002.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2001.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse2000.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse99.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse98.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse97.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse96.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse95.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse94.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse93.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse92.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse91.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse90.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse89.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse88.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse87.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse86.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse85.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse84.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse83.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse82.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse81.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse80.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse79.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse78.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse77.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/icse76.bht%3A&h=1000&format=json'
    ],
    'FairWare@ICSE': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/icse/fairware2018.bht%3A&h=1000&format=json'
    ],
    'ISSTA': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2016.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2015.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2014.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2013.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2012.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2011.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2010.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2009.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2008.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2007.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2006.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2005.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2004.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2003.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2002.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2001.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta2000.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta98.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta96.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta94.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/issta93.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/tav91.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/issta/tav89.bht%3A&h=1000&format=json'
    ],
    'ASE': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2016.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2015.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2014.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2013.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2012.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2011.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2010.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2009.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2008.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2007.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2006.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2005.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2004.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2003.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2002.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2001.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase2000.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase1999.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase1998.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/ase1997.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/kbse1996.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/kbse1995.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/kbse1994.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/kbse1993.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/kbse1992.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/kbse/kbse1991.bht%3A&h=1000&format=json'
    ],

    'ACL': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2019-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2018-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2017-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2016-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2018-2.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2017-2.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/acl/acl2016-2.bht%3A&h=1000&format=json'
    ],
    'EMNLP': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/emnlp/emnlp2020-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/emnlp/emnlp2019-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/emnlp/emnlp2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/emnlp/emnlp2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/emnlp/emnlp2016.bht%3A&h=1000&format=json'
    ],
    'AAAI': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/aaai/aaai2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/aaai/aaai2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/aaai/aaai2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/aaai/aaai2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/aaai/aaai2016.bht%3A&h=1000&format=json'
    ],
    'CVPR': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/cvpr/cvpr2020.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/cvpr/cvpr2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/cvpr/cvpr2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/cvpr/cvpr2017.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/cvpr/cvpr2016.bht%3A&h=1000&format=json'
    ],
    'ICCV': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/iccv/iccv2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/iccv/iccv2017.bht%3A&h=1000&format=json'
    ],
    'ECCV': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-2.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-3.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-4.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-5.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-6.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-7.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-8.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-9.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-10.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-11.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-12.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-13.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-14.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-15.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-16.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-17.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-18.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-19.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-20.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-21.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-22.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-23.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-24.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-25.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-26.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-27.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-28.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-29.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2020-30.bht%3A&h=1000&format=json',

        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-2.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-3.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-4.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-5.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-6.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-7.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-8.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-9.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-10.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-11.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-12.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-13.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-14.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-15.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2018-16.bht%3A&h=1000&format=json',

        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-1.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-2.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-3.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-4.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-5.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-6.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-7.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/eccv/eccv2016-8.bht%3A&h=1000&format=json'
    ],
    'FAT': [
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/fat/fat2018.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/fat/fat2019.bht%3A&h=1000&format=json',
        'https://dblp.org/search/publ/api?q=toc%3Adb/conf/fat/fat2020.bht%3A&h=1000&format=json'
    ]
}


def normalize_word(s):
    stemmer = nltk.stem.SnowballStemmer('english')
    s = re.sub('\W+', '', s)
    s = s.lower()
    s = stemmer.stem(s)

    return s

def title_match(title, keywords):
    title_list = [normalize_word(e.strip()) for e in title.split(' ')]
    for keyword in keywords:
        if normalize_word(keyword) in title_list:
            return True 
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    args = parser.parse_args()

    conf = args.conf
    link_list = links[conf]
        
    for i, conf_link in enumerate(link_list):
        print("Start:", conf_link)
        paper_dict = web_utils.d = web_utils.get_dblp_venue(conf_link)
        p = Path('./dump', conf)
        p.mkdir(exist_ok=True, parents=True)
        with open(str(Path(p, str(i) + '.yaml')), 'w') as f:
            yaml.safe_dump(paper_dict, f)
            

        


if __name__ == "__main__":
    main()
