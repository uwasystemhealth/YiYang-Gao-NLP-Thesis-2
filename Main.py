import Utility
import Custom_Words
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    print('----------------------------------------------------------------')
    print('starting pipeline')
    print(Custom_Words.Custom_Words.stop_words)
    print(Custom_Words.Custom_Words.failure_noun)
    print(Custom_Words.Custom_Words.positional_words)
