#%%
import streamlit as st
import tweepy
#%%
def benchmark(function):
    import time
    def wrapper(*args):
        start = time.time()
        result = function(*args)
        end = time.time()
        st.write(f'[*] Время выполнения {function} {end - start:.2E} секунд')
        return result
    return wrapper

#%%
def long_running(function):
    def wrap(*args, **kwargs):
        placeholder = st.empty()
        placeholder.text(f"Функция {function.__name__} может выполняться достаточно долго...")
        result = function()
        placeholder.empty()
        return result
    return wrap

#%%
def error_wrap_auth(auth_func):
    def wrap(*args, **kwargs):
        tries = 0
        try:
            api = auth_func()
            # st.write(api)
            return api
        except tweepy.TweepyException:
            if tries < 3:
                tries += 1
                api = auth_func()
                return api
            else:
                st.write("Возникла нерешаемая ошибка с подключением к твиттеру")
    return wrap

#%%
if __name__ == '__main__':
    print("Это вспомогательный модуль и он не должен исполняться самостоятельно")