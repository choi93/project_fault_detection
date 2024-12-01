from segFault.segFault import Trainer


if __name__ == "__main__":
    trainer = Trainer('segFault/config/train_config.yaml')
    trainer.train() 
