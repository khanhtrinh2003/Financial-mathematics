
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import datetime


class Stochastic_simulation():
    def __init__(self,close:pd.DataFrame,region='vn',universe='TOP500',delay=1) -> None:
        self.region=region
        self.universe=universe
        self.delay=delay
        self.close=close
    
        self.returns=self.close.pct_change()

    def test_for_normal(self):
        from scipy.stats import jarque_bera
        p_value_dict={}
        
        for i in self.returns.columns:
            try:
                jb_value, p_value = jarque_bera(self.returns[i].dropna())
                p_value_dict[i]=[jb_value,p_value]
            except:
                pass
        p_df=pd.DataFrame(p_value_dict).T
        p_df.columns=['jb_value','p_value']
        p_df.sort_values('p_value',inplace=True)
        return p_df
    
    def plot_box_returns(self,symbols:list):
        fig = px.box(self.returns[symbols],title='Returns')
        fig.show()
        
    def describe_returns(self):
        return self.returns.describe()

    def mu_sigma(self,df,dt):
        mu = (df.pct_change().mean())/dt
        sigma = np.sqrt(df.pct_change().var()/dt)
        return mu, sigma

    def get_mu_sigma(self):
        mu_dict={}
        for ticker in self.close.columns: 
            data = self.close[ticker].dropna()
            T=1
            so_buoc=len(data) # Số bước 
            dt=T/so_buoc
            mu, sigma = self.mu_sigma(data,dt)  
            mu_dict[ticker]=[mu,sigma]
        p_df=pd.DataFrame(mu_dict).T
        p_df.columns=['muy','sigma']
        p_df.sort_values('muy',ascending=False,inplace=True)
        return p_df

    def estimate_days(self,ticker,price0,price1):
        '''
        Estimate number of days to reach price1 from price0
        '''
        data = self.close[ticker].dropna()
        T=1
        so_buoc=len(data) # Số bước 
        dt=T/so_buoc

        t=(np.cumsum(np.ones(so_buoc))-1)/(so_buoc-1)

        mu, sigma = self.mu_sigma(data,dt)
        if np.sign(mu)*np.sign(price1-price0)<=0:
            return f'Cannot reach {price1} from {price0}'
        
        days=int(np.log(price1/price0)/(mu-0.5*(sigma**2))*(so_buoc-1)+1)

        def convert_days(days):
            if days < 0:
                raise ValueError("Input should be a non-negative integer.")

            years = days // 365
            days -= years * 365

            months = days // 30
            days -= months * 30

            result = ""
            if years > 0:
                result += f"{years} year{'s' if years > 1 else ''} "

            if months > 0:
                result += f"{months} month{'s' if months > 1 else ''} "

            if days > 0:
                result += f"{days} day{'s' if days > 1 else ''}"

            return result.strip()        
        
        return convert_days(days)
    
    def plot_price_simulation(self,ticker,so_kich_ban=3):
        data = self.close[ticker].dropna()
        T=1
        so_buoc=len(data) # Số bước 
        dt=T/so_buoc

        t=(np.cumsum(np.ones(so_buoc))-1)/(so_buoc-1)

        mu, sigma = self.mu_sigma(data,dt)

        W = np.cumsum(np.sqrt(dt)*pd.DataFrame(np.random.randn(so_buoc,so_kich_ban)).shift(1).fillna(0))
        S0=data.iloc[0,]
        U=S0*np.exp(pd.DataFrame([((mu-0.5*sigma**2)*t)]*so_kich_ban).T+sigma*W)

        U['Price_motion'] =S0*pd.DataFrame(np.exp((mu-0.5*sigma**2)*t))
        U['Real_Price']=data.values
        U.index=data.index
        fig = px.line(U, x=U.index, y=U.columns, title=f'{ticker}')
        fig.show()           
        print(f'mu: {mu}')
        print(f'sigma: {sigma}')    

def convert_timestamp(x):
    return datetime.datetime.fromtimestamp(x['TradingDate']).strftime('%Y-%m-%d')

class My_db():
    def __init__(self,database) -> None:
        self.database=database
        
        
    def get_database_table(self):
        
        self.cursor= self.database.cursor()
        # Execute the query to get the table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';") # replace with appropriate query for your database

        # Fetch all the table names
        tables = self.cursor.fetchall()

        # Print the table names
        for table in tables:
            print(table[0])

        # Commit the changes
        self.database.commit()

        # Close the self.cursor and connection
        self.cursor.close()

    def delete_database_table(self,name):
        # Get the self.cursor
        
        self.cursor= self.database.cursor()
        # Execute the query to delete the table
        self.cursor.execute(f"DROP TABLE IF EXISTS {name};") # replace "table_name" with the name of the table you want to delete

        # Commit the changes
        self.database.commit()

        # Close the self.cursor and connection
        self.cursor.close()

    def get_shape_database(self):
        
        self.cursor= self.database.cursor()
        # Get the table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        # Iterate through the tables and get their shape
        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            num_rows = self.cursor.fetchone()[0]
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            num_cols = len(self.cursor.fetchall())
            print(f"{table_name}: {num_rows} rows, {num_cols} columns")

        # Close the self.cursor and connection
        self.cursor.close()   

    def read_database_tabe(self, table) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {table}", self.database)
    
    def read_all(self,data):
        cl = self.read_database_tabe(data)
        cl.rename(columns={'Date':'TradingDate','time':'TradingDate','OPEN_TIME':'TradingDate'},inplace=True)
        try:
            cl['TradingDate']=cl.apply(convert_timestamp,axis=1)
        except:
            pass
        cl['TradingDate'] = pd.to_datetime(cl['TradingDate'])
        cl.set_index("TradingDate",inplace=True)    
        return cl    
    
    def read_data(self,data,universe:list):
        cl = self.read_all(data) 
        try:
            cl[np.setdiff1d(universe,cl.columns)]=np.NaN
        except:
            pass 
        return cl[universe]

    def stock_price(self,universe:list):
        close = self.read_data("close",universe)
        open = self.read_data("open",universe)
        high = self.read_data("high",universe)
        low = self.read_data("low",universe)
        volume = self.read_data("volume",universe)
        returns = close.pct_change()
        return close, open, high, low, volume, returns
    
    def save_database_tabe(self,df,table,mode="append"):
        df.to_sql(f"{table}",con=self.database,if_exists=mode,index=False)  
    
    def close_db(self):
        self.cursor.close()  