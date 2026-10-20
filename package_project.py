"""Package source, frozen split files, weights and evidence, excluding runtimes."""
import os,zipfile
from pathlib import Path
R=Path(__file__).parent.resolve()
dest=R.parent/'RNN_Project_Zouhair_Eouichak.zip'
allowed_dirs={'assets','logs','output','results','screenshots'}
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in R.iterdir():
        if p.is_file() and (p.suffix in {'.py','.ps1','.md','.ipynb','.txt'} or p.name in {'.gitignore','.gitattributes'}):
            z.write(p,'rnn_drug_review/'+p.name)
        elif p.is_dir() and p.name in allowed_dirs:
            for base,dirs,files in os.walk(p):
                for name in files:
                    f=Path(base)/name
                    if name.startswith('00_'):continue
                    z.write(f,'rnn_drug_review/'+f.relative_to(R).as_posix())
    for source,destination in [('.build/build_slides.mjs','tools/build_slides.mjs'),('.build/slides-validation.json','validation/slides-validation.json')]:
        f=R/source
        if f.exists():z.write(f,'rnn_drug_review/'+destination)
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    assert any(n.endswith('results/cifg.pt') for n in z.namelist())
    assert any(n.endswith('output/RNN_Presentation.pptx') for n in z.namelist())
print(dest, dest.stat().st_size)
