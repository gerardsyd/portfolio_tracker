"""Check all imports in project match what's in requirements.txt."""
import ast
from pathlib import Path

files = list(Path('app').rglob('*.py')) + list(Path('utils').rglob('*.py'))
files += [Path('config.py'), Path('main.py'), Path('wsgi.py')]

imports = set()
for f in files:
    try:
        raw = f.read_bytes()
        if raw[:3] == b'\xef\xbb\xbf':
            raw = raw[3:]
        tree = ast.parse(raw)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f'FAILED: {f}: {e}')

req_lines = Path('requirements.txt').read_text().strip().splitlines()
reqs = set()
for line in req_lines:
    line = line.strip()
    if line and not line.startswith('#'):
        pkg = line.split('==')[0].split('>=')[0].split('<')[0].split('[')[0].strip()
        if pkg: reqs.add(pkg)

# Standard library
stdlib = {
    'os','sys','re','json','math','datetime','logging','typing','io','warnings',
    'collections','functools','copy','traceback','itertools','pathlib','ast',
    'urllib','abc','operator','threading','time','enum','hashlib','uuid','csv',
    'string','textwrap','types','subprocess','tempfile','configparser','argparse',
    'base64','socket','ssl','glob','html','http','importlib','inspect','pickle',
    'subprocess','random','secrets','signal','stat','token','trace','unittest',
    'zipfile','zoneinfo','numbers','bisect','calendar','decimal','difflib',
    'fnmatch','fractions','getopt','textwrap','locale','queue','reprlib','struct',
    'tarfile','webbrowser','dis','pprint',
}

local = {'app','utils','config','main','wsgi','migrations','pftracker'}

# Map import name -> PyPI package name
pypi = {
    'flask': 'flask','werkzeug':'flask','jinja2':'flask','click':'flask',
    'markupsafe':'flask','itsdangerous':'flask','blinker':'flask',
    'flask_login':'flask-login','flask_sqlalchemy':'flask-sqlalchemy',
    'flask_migrate':'flask-migrate','flask_wtf':'flask-wtf','wtforms':'flask-wtf',
    'dotenv':'python-dotenv','sqlalchemy':'sqlalchemy','alembic':'alembic',
    'wtforms':'wtforms',
    'numpy':'numpy','pandas':'pandas','scipy':'scipy',
    'yfinance':'yfinance','binance':'python-binance',
    'plotly':'plotly','gunicorn':'gunicorn',
    'cryptography':'cryptography',
    'requests':'requests','bs4':'beautifulsoup4','lxml':'lxml','selenium':'selenium',
    'dateutil':'python-dateutil','tzlocal':'tzlocal','pytz':'pytz',
    'email_validator':'email-validator','tenacity':'tenacity','packaging':'packaging',
    'mysqlclient':'mysqlclient','pymysql':'pymysql','xlsxwriter':'xlsxwriter',
    'pyopenssl':'pyopenssl','certifi':'certifi','pycparser':'pycparser',
    'numpy':'numpy','tornado':'tornado',
}

missing = []
for imp in sorted(imports):
    if imp in stdlib or imp in local:
        continue
    pkg = pypi.get(imp, imp)
    if pkg not in reqs:
        missing.append(f'  import "{imp}" -> PyPI "{pkg}"')

if missing:
    print('=== MISSING PACKAGES ===')
    for m in missing:
        print(m)
else:
    print('=== ALL IMPORTS ACCOUNTED FOR ===')
