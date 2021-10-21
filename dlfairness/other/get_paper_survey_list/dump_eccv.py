import yaml
from pathlib import Path 

import web_utils

papers_2020, papers_2018 = web_utils.eccv_2020_2018()


p = Path('./dump', 'ECCV')
p.mkdir(exist_ok=True, parents=True)
with open(str(Path(p, '0.yaml')), 'w') as f:
    yaml.safe_dump(papers_2020, f)

with open(str(Path(p, '1.yaml')), 'w') as f:
    yaml.safe_dump(papers_2018, f)