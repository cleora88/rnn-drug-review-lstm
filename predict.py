"""Run a saved model on a review. Ratings are not input features."""
import argparse, json
from pathlib import Path
import torch
from models import Classifier
from run import clean

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('text')
    parser.add_argument('--model',choices=['vanilla','cifg','wmc'],default='cifg')
    args=parser.parse_args()
    root=Path(__file__).parent/'results'
    vocab=json.loads((root/'vocabulary.json').read_text())
    model=Classifier(len(vocab),args.model)
    model.load_state_dict(torch.load(root/f'{args.model}.pt',map_location='cpu',weights_only=True));model.eval()
    tokens=clean(args.text).split()[:96]
    x=torch.tensor([[vocab.get(t,1) for t in tokens]+[0]*(96-len(tokens))])
    with torch.no_grad(): probabilities=model(x).softmax(-1)[0].tolist()
    print(json.dumps(dict(zip(['negative','neutral','positive'],probabilities)),indent=2))
    print('Probabilities come from class-weighted training and are not calibrated confidence estimates.')
if __name__=='__main__':main()
