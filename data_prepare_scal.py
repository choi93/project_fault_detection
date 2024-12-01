from segFault.Data2Patch import Data2patch

def main():
    d2p = Data2patch(config_path='segFault/config/data_preprocess_config.yaml')  # config 경로 지정
    d2p.run()

if __name__ == '__main__':
    main()



