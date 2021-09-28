import os
import numpy as np
import pandas as pd
import datetime
import time
import random


############################
## Constants and Parameters
############################

TOTAL_CUSTOMER_COUNT = 5000
TOTAL_TERMINAL_COUNT = 2500
TOTAL_SIMULATION_DAYS = 365
SIMULATION_START_DATE = '2018-01-01'

CUSTOMER_ACTIVITY_RADIUS = 30

# Random seeds (arbitrary)
GLOBAL_RANDOM_SEED = 47
CUSTOMER_PROFILE_GENERATION_RANDOM_SEED = 19700 + GLOBAL_RANDOM_SEED
TERMINAL_PROFILE_GENERATION_RANDOM_SEED = 20010 + GLOBAL_RANDOM_SEED
GENUINE_TRANSACTION_GENERATION_RANDOM_SEED = 18810 + GLOBAL_RANDOM_SEED
DEFAULT_FRAUD_GENERATION_RANDOM_SEED = 19430 + GLOBAL_RANDOM_SEED
LARGE_AMOUNT_FRAUD_GENERATION_RANDOM_SEED = 11 * 2036 + GLOBAL_RANDOM_SEED
CARD_TEST_FRAUD_GENERATION_RANDOM_SEED = 13 * 123 + GLOBAL_RANDOM_SEED
SUCCESSIVE_SMALL_AMOUNT_FRAUD_GENERATION_RANDOM_SEED = 13 * 9876 + GLOBAL_RANDOM_SEED
MERCHANT_COLLUSION_FRAUD_GENERATION_RANDOM_SEED = 13 * 1024 + GLOBAL_RANDOM_SEED

# Constants for fraud patterns
FRAUD_PATTERN_2 = 12        # fraud pattern 2 as in handbook
FRAUD_PATTERN_3 = 13        # fraud pattern 3 as in handbook
FRAUD_PATTERN_LA = 21       # One or more large amount
FRAUD_PATTERN_CT = 22       # Card test followed by large amount
FRAUD_PATTERN_SA = 23       # Successive small amount
FRAUD_PATTERN_MC = 24       # Merchant collusion


# Constants for fraud pattern LA (one or more large amount)
FRAUD_PATTERN_LA_CASE_PERCENTAGE = 0.08                                         # no. of compromised customer / no. of genuine transaction * 100
FRAUD_PATTERN_LA_SUCCESSIVE_PROBABILITY = 0.25                                  # Probability of being able to conduct fradulent transaction a second (third, forth, etc) time
FRAUD_PATTERN_LA_AMOUNT_RANGE = (50,200)                                     # Fradulent transaction amount range

# Constants for fraud pattern CT (card test then large amount)
FRAUD_PATTERN_CT_CASE_PERCENTAGE = 0.08                                         # no. of compromised customer / no. of genuine transaction * 100
FRAUD_PATTERN_CT_SUCCESSIVE_PROBABILITY1 = 0.8                                   # Probability of being able to conduct fradulent transaction after a card test
FRAUD_PATTERN_CT_SUCCESSIVE_PROBABILITY2 = 0.3                                   # Probability of being able to conduct fradulent transaction again
FRAUD_PATTERN_CT_AMOUNT_RANGE = (90,210)                                     # Fradulent transaction amount range
FRAUD_PATTERN_CT_CARD_TEST_TERMINAL_COUNT = max(1, round(TOTAL_TERMINAL_COUNT * 0.01))  # No of terminals that may be used for card test

# Constants for fraud pattern SA (successive small amount)
FRAUD_PATTERN_SA_CASE_PERCENTAGE = 0.1                                         # no. of compromised customer / no. of genuine transaction * 100
FRAUD_PATTERN_SA_SUCCESSIVE_PROBABILITY = 0.7                                   # Probability of being able to conduct fradulent transaction again
FRAUD_PATTERN_SA_AMOUNT_RANGE = (50,100)                                        # Fradulent transaction amount range
FRAUD_PATTERN_SA_INTERVAL_RANGE = (60 * 60, 60 * 60 * 24)                       # Number of seconds between frauds

# Constants for fraud pattern MC (merchant collusion)
FRAUD_PATTERN_MC_CASE_PERCENTAGE = 0.08                                          # no. of compromised customer / no. of genuine transaction * 100
FRAUD_PATTERN_MC_SUCCESSIVE_PROBABILITY = 0.4                                   # Probability of being able to conduct fradulent transaction again
FRAUD_PATTERN_MC_AMOUNT_RANGE = (60,220)                                     # Fradulent transaction amount range
FRAUD_PATTERN_MC_COLLUDED_MERCHANT_COUNT = max(1, round(TOTAL_TERMINAL_COUNT * 0.08))   # Number of colluded merchants

#################################
## Output path and file  names
#################################

DIR_OUTPUT = "./simulated-data/"
FULLSET_FILENAME = "trans.csv"
CUSTOMER_PROFILE_FILENAME = "profile_customer.csv"
TERMINAL_PROFILE_FILENAME = "profile_terminal.csv"




############################################################################################################################
## Supporting functions adopted from:
##   https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html
############################################################################################################################


def generate_customer_profiles_table(n_customers, random_state=0):
    
    np.random.seed(random_state)
        
    customer_id_properties=[]
    
    # Generate customer properties from random distributions 
    for customer_id in range(n_customers):
        
        x_customer_id = np.random.uniform(0,100)
        y_customer_id = np.random.uniform(0,100)
        
        mean_amount = np.random.uniform(5,100) # Arbitrary (but sensible) value 
        std_amount = mean_amount/2 # Arbitrary (but sensible) value
        
        mean_nb_tx_per_day = np.random.uniform(0,4) # Arbitrary (but sensible) value 
        
        customer_id_properties.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_nb_tx_per_day])
        
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                      'x_customer_id', 'y_customer_id',
                                                                      'mean_amount', 'std_amount',
                                                                      'mean_nb_tx_per_day'])
    
    return customer_profiles_table


def generate_terminal_profiles_table(n_terminals, random_state=0):
    
    np.random.seed(random_state)
        
    terminal_id_properties=[]
    
    # Generate terminal properties from random distributions 
    for terminal_id in range(n_terminals):
        
        x_terminal_id = np.random.uniform(0,100)
        y_terminal_id = np.random.uniform(0,100)
        
        terminal_id_properties.append([terminal_id,
                                      x_terminal_id, y_terminal_id])
                                       
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                      'x_terminal_id', 'y_terminal_id'])
    
    return terminal_profiles_table


def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    
    # Use numpy arrays in the following to speed up computations
    
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y<r)[0])
    
    # Return the list of terminal IDs
    return available_terminals
    

def generate_transactions_table(customer_profile, start_date = "2018-04-01", nb_days = 10):
    
    c_id = customer_profile.CUSTOMER_ID

    customer_transactions = []
    
    random.seed(customer_profile.CUSTOMER_ID + GENUINE_TRANSACTION_GENERATION_RANDOM_SEED)
    np.random.seed(int(customer_profile.CUSTOMER_ID + GENUINE_TRANSACTION_GENERATION_RANDOM_SEED))
    
    # For all days
    for day in range(nb_days):
        
        # Random number of transactions for that day 
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        
        # If nb_tx positive, let us generate transactions
        if nb_tx>0:
            
            for tx in range(nb_tx):
                
                # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that 
                # most transactions occur during the day.
                time_tx = int(np.random.normal(86400/2, 20000))
                
                # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
                if (time_tx>0) and (time_tx<86400):
                    
                    # Amount is drawn from a normal distribution  
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    
                    # If amount negative, draw from a uniform distribution
                    if amount <= 0:
                        amount = np.random.uniform(0,customer_profile.mean_amount*2)
                    
                    # If amount is less than one, add one to make it more realistic
                    if amount < 1:
                        amount += 1

                    amount=np.round(amount,decimals=2)
                    
                    if len(customer_profile.available_terminals)>0:
                        
                        terminal_id = random.choice(customer_profile.available_terminals)
                    
                        customer_transactions.append([time_tx+day*86400, day,
                                                      customer_profile.CUSTOMER_ID, 
                                                      terminal_id, amount])

    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    
    if len(customer_transactions)>0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions=customer_transactions[['TX_DATETIME','CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    else:
        print('No transaction for customer', customer_profile.CUSTOMER_ID)

    return customer_transactions  
    
    
def generate_dataset(n_customers = 10000, n_terminals = 1000000, nb_days=90, start_date="2018-04-01", r=5):
    
    start_time=time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = CUSTOMER_PROFILE_GENERATION_RANDOM_SEED)
    print("Time to generate customer profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = TERMINAL_PROFILE_GENERATION_RANDOM_SEED)
    print("Time to generate terminal profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    # With Pandarallel
    #customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_closest_terminals(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals']=customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').apply(lambda x : generate_transactions_table(x.iloc[0], start_date=start_date, nb_days=nb_days)).reset_index(drop=True)
    # With Pandarallel
    #transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').parallel_apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2}s".format(time.time()-start_time))
    
    # Sort transactions chronologically
    transactions_df=transactions_df.sort_values('TX_DATETIME')
    
    # These transactions are all genuine
    transactions_df['TX_FRAUD']=0
    transactions_df['TX_FRAUD_SCENARIO']=0

    return (customer_profiles_table, terminal_profiles_table, transactions_df)


# Default fraud patterns from handbook (with secarario 1 removed)
def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df, start_date):
    
    '''
    # Scenario 1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD']=1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD_SCENARIO']=1
    nb_frauds_scenario_1=transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1: "+str(nb_frauds_scenario_1))
    '''

    # Scenario 2
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state = day + DEFAULT_FRAUD_GENERATION_RANDOM_SEED)
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+10) & 
                                                    (transactions_df.TERMINAL_ID.isin(compromised_terminals))]
                            
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD']=1
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD_SCENARIO'] = FRAUD_PATTERN_2
    
    nb_frauds_scenario_2=transactions_df.TX_FRAUD.sum() # -nb_frauds_scenario_1
    print("Number of frauds from scenario 2: "+str(nb_frauds_scenario_2))
    
    # Scenario 3
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state = day + DEFAULT_FRAUD_GENERATION_RANDOM_SEED).values
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+14) & 
                                                    (transactions_df.CUSTOMER_ID.isin(compromised_customers))]
        
        nb_compromised_transactions=len(compromised_transactions)
        
        
        random.seed(day + DEFAULT_FRAUD_GENERATION_RANDOM_SEED)
        index_fauds = random.sample(list(compromised_transactions.index.values),k=int(nb_compromised_transactions/3))
        
        transactions_df.loc[index_fauds,'TX_AMOUNT'] = transactions_df.loc[index_fauds,'TX_AMOUNT']*5
        transactions_df.loc[index_fauds,'TX_FRAUD'] = 1
        transactions_df.loc[index_fauds,'TX_FRAUD_SCENARIO'] = FRAUD_PATTERN_3


    nb_frauds_scenario_3=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_2 # -nb_frauds_scenario_1
    print("Number of frauds from scenario 3: "+str(nb_frauds_scenario_3))
    
    '''  produce incorrect result... give up for now
    # change the time of scenario 3
    cnt = 0
    for i, tr in transactions_df.iterrows():
        if cnt % 5000 == 0:
            print('cnt  ',  cnt, ' i ', i)

        cnt += 1

        if tr.TX_FRAUD_SCENARIO == FRAUD_PATTERN_3:
            cust_id = tr.CUSTOMER_ID
            amt = tr.TX_AMOUNT
            term_id = tr.TERMINAL_ID
            f_day = tr.TX_TIME_DAYS
            f_tm = round(f_day * 86400 + (86400/24 * 3))    # this is the number of seconds elapsed since T=0 to T=f_day at 3am
            f_v = round(np.random.normal(0, 20000))
            f_tm += f_v  # now f_tm is the number of seconds elapsed since T=0 to the time a fraud will be committed
            f_tm = max(0, f_tm) #  rare case
            tx_dt = pd.to_datetime(f_tm, unit='s', origin=start_date)
            transactions_df.loc[i, 'TX_TIME_SECONDS'] = f_tm
            transactions_df.loc[i, 'TX_DATETIME'] = tx_dt
    '''


    
    return transactions_df                 


#######################################################################################
## Fraud pattern :
##   One or more large amount
#######################################################################################

class LargeAmountFraudster:

    def generate_transactions(self, customer_ids, terminal_ids, case_count, successive_probability, amount_range, nb_days, random_state = 0):
        np.random.seed(random_state)
        victim_ids = np.random.choice(list(customer_ids), case_count, replace = True)

        trans = []
        terminal_ids = list(terminal_ids)

        for victim_id in victim_ids:
            target_terminal_id = np.random.choice(terminal_ids)
            t = self.__commit_fraud(victim_id, target_terminal_id, successive_probability, amount_range, nb_days)
            trans += t

        return trans

    def __commit_fraud(self, victim_id, terminal_id, successive_probability, amount_range, nb_days):
        # Pick a day and time to commit fraud

        f_day = np.random.choice(range(0, nb_days))     # pick a day
        f_tm = round(f_day * 86400 + (86400/24 * 2))    # this is the number of seconds elapsed since T=0 to T=f_day at 2am
        f_v = round(np.random.normal(0, 20000))
        f_tm += f_v  # now f_tm is the number of seconds elapsed since T=0 to the time a fraud will be committed
        f_tm = max(0, f_tm) #  rare case

        trans = []
        alerted = False
        succ_proba = successive_probability

        while not alerted:
            amt = np.random.uniform(*amount_range)
            amt = round(amt)    # large amount transactions are usually rounded in dollars
            
            trans.append((f_tm, f_tm // 86400, victim_id, terminal_id, amt))

            # chance of committing another fraud?
            alerted = np.random.choice([0, 1], p = [succ_proba, 1-succ_proba])
            succ_proba = round(succ_proba * succ_proba, 2)
            # commit next fraud within 10-30 mins if not alerted, using same terminal
            f_tm += round(np.random.uniform(10, 30)) * 60

        return trans

#######################################################################################
## Fraud pattern :
##   Card test followed by large amount
#######################################################################################

class CardTestFraudster:

    def generate_transactions(self, customer_ids, terminal_ids, fraud_count, test_terminal_count, fraud_probability, successive_probability, amount_range, nb_days, random_state = 0):
        np.random.seed(random_state)
        victim_ids = np.random.choice(list(customer_ids), fraud_count, replace = True)
        terminal_ids = list(terminal_ids)
        test_terminal_ids = np.random.choice(terminal_ids, test_terminal_count, replace = True)

        trans = []

        for victim_id in victim_ids:
            test_terminal_id = np.random.choice(test_terminal_ids)
            target_terminal_id = np.random.choice(terminal_ids)
            t = self.__commit_fraud(victim_id, test_terminal_id, target_terminal_id, fraud_probability, successive_probability, amount_range, nb_days)
            trans += t

        return trans

    def __commit_fraud(self, victim_id, test_terminal_id, terminal_id, fraud_probability, successive_probability, amount_range, nb_days):
        # Pick a day and time to commit fraud

        f_day = np.random.choice(range(0, nb_days))     # pick a day
        f_tm = round(f_day * 86400 + (86400/24 * 3))    # this is the number of seconds elapsed since T=0 to T=f_day at 3am
        f_v = round(np.random.normal(0, 20000))
        f_tm += f_v  # now f_tm is the number of seconds elapsed since T=0 to the time a fraud will be committed
        f_tm = max(0, f_tm) #  rare case

        trans = []
        alerted = False

        while not alerted:
            if len(trans) == 0:
                # card test
                term_id = test_terminal_id
                amt = 1
                # chance of continuing to commit fraud?
                succ_proba = fraud_probability
                alerted = np.random.choice([0, 1], p = [succ_proba, 1-succ_proba])
                succ_proba = successive_probability
            else:
                term_id = terminal_id
                amt = np.random.uniform(*amount_range)
                amt = round(amt)    # large amount transactions are usually rounded in dollars
                # chance of continuing to commit fraud?
                alerted = np.random.choice([0, 1], p = [succ_proba, 1-succ_proba])
                succ_proba = round(succ_proba * succ_proba, 2)
            
            trans.append((f_tm, f_tm // 86400, victim_id, term_id, amt))

            # commit next fraud within 10-30 mins if not alerted, using same terminal
            f_tm += round(np.random.uniform(10, 20)) * 60

        return trans


#######################################################################################
## Fraud pattern :
##   Successive small amount
#######################################################################################

class SuccessiveSmallAmountFraudster:

    def generate_transactions(self, customer_ids, terminal_ids, fraud_count, successive_probability, interval_range, amount_range, nb_days, random_state = 0):
        np.random.seed(random_state)
        victim_ids = np.random.choice(list(customer_ids), fraud_count, replace = True)
        self.__terminal_ids = list(terminal_ids)

        trans = []

        for victim_id in victim_ids:
            t = self.__commit_fraud(victim_id, successive_probability, interval_range, amount_range, nb_days)
            trans += t

        return trans

    def __commit_fraud(self, victim_id, successive_probability, interval_range, amount_range, nb_days):
        # Pick a day and time to commit fraud

        f_day = np.random.choice(range(0, nb_days))     # pick a day
        f_tm = round(f_day * 86400 + (86400/24 * 1))    # this is the number of seconds elapsed since T=0 to T=f_day at 1am
        f_v = round(np.random.normal(0, 20000))
        f_tm += f_v  # now f_tm is the number of seconds elapsed since T=0 to the time a fraud will be committed
        f_tm = max(0, f_tm) #  rare case

        trans = []
        alerted = False
        succ_proba = successive_probability

        while not alerted:
            term_id = np.random.choice(self.__terminal_ids)
            amt = np.random.uniform(*amount_range)
            amt = round(amt, 2)
            
            trans.append((f_tm, f_tm // 86400, victim_id, term_id, amt))

            # chance of continuing to commit fraud?
            alerted = np.random.choice([0, 1], p = [succ_proba, 1-succ_proba])
            succ_proba = round(succ_proba * 0.8, 2)
            # commit next fraud within the interval range if not alerted
            f_tm += round(np.random.uniform(*interval_range))

        return trans

#######################################################################################
## Fraud pattern :
##   Merchant collusion
#######################################################################################

class MerchantCollusionFraudster:

    def generate_transactions(self, customer_profile_table, terminal_ids, fraud_count, colluded_merchant_count, successive_probability, amount_range, nb_days, random_state = 0):
        np.random.seed(random_state)
        victim_profiles = customer_profile_table.sample(n = fraud_count, replace = True)
        self.__terminal_ids = list(terminal_ids)
        colluded_terminal_ids = np.random.choice(list(terminal_ids), colluded_merchant_count)

        trans = []

        for _, victim_profile in victim_profiles.iterrows():
            colluded_terminal_id = np.random.choice(colluded_terminal_ids)
            t = self.__commit_fraud(victim_profile, colluded_terminal_id, successive_probability, amount_range, nb_days)
            trans += [t]

        return trans

    def __commit_fraud(self, victim_profile, colluded_terminal_id, successive_probability, amount_range, nb_days):
        # Pick a day and time for customer to purchase in colluded merchant

        f_day = np.random.choice(range(0, nb_days))     # pick a day
        f_tm = round(f_day * 86400 + (86400/24 * 12))   # this is the number of seconds elapsed since T=0 to T=f_day at 12pm
        f_v = round(np.random.normal(0, 20000))
        f_tm += f_v  # now f_tm is the number of seconds elapsed since T=0 to the time of purchase
        f_tm = max(0, f_tm) #  rare case

        trans = []

        victim_id = victim_profile.CUSTOMER_ID

        # Create a genuine transaction
        amt = np.random.normal(victim_profile.mean_amount, victim_profile.std_amount)
        # If amount negative, draw from a uniform distribution
        if amt <= 0:
            amt = np.random.uniform(0,victim_profile.mean_amount*2)
        amt = np.round(amt, decimals = 2)
        if amt < 1:
            amt += 1
        trans.append((f_tm, f_tm // 86400, victim_id, colluded_terminal_id, amt))

        # Commit fraud some time later (7 - 14 days)
        f_day = (f_tm // 86400) + round(np.random.uniform(7, 14))     # pick a day
        f_tm = round(f_day * 86400 + (86400/24 * 2))   # this is the number of seconds elapsed since T=0 to T=f_day at 2am
        f_v = round(np.random.normal(0, 20000))
        f_tm += f_v  # now f_tm is the number of seconds elapsed since T=0 to the time of fraud
        f_tm = max(0, f_tm) #  rare case

        alerted = False
        succ_proba = successive_probability

        while not alerted:
            amt = np.random.uniform(*amount_range)
            amt = round(amt)    # large amount transactions are usually rounded in dollars
            target_terminal_id = np.random.choice(self.__terminal_ids)
            
            trans.append((f_tm, f_tm // 86400, victim_id, target_terminal_id, amt))

            # chance of committing another fraud?
            alerted = np.random.choice([0, 1], p = [succ_proba, 1-succ_proba])
            succ_proba = round(succ_proba * succ_proba, 2)
            # commit next fraud within 10-30 mins if not alerted
            f_tm += round(np.random.uniform(10, 30)) * 60

        return trans

###########################
## Program entry point 
###########################

def main():

    print('Customer count', TOTAL_CUSTOMER_COUNT)
    print('Terminal count', TOTAL_TERMINAL_COUNT)
    print('Simulation days', TOTAL_SIMULATION_DAYS)

    # Generate genuine transactions
    (customer_profiles_table, terminal_profiles_table, transactions_df) = \
        generate_dataset(
            n_customers = TOTAL_CUSTOMER_COUNT, 
            n_terminals = TOTAL_TERMINAL_COUNT, 
            nb_days = TOTAL_SIMULATION_DAYS, 
            start_date = SIMULATION_START_DATE, 
            r = CUSTOMER_ACTIVITY_RADIUS
        )

    # No of genuine transactions
    genuine_transaction_count = transactions_df.shape[0]
    print('No of genuien transactions', genuine_transaction_count)

    # Add default frauds (scenario 2 & 3 from handbook)      
    transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df, SIMULATION_START_DATE)

    customer_ids = set(customer_profiles_table.CUSTOMER_ID)
    terminal_ids = set(terminal_profiles_table.TERMINAL_ID)

    # More fraudulent transactions,  array of ('TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT')
    more_fraud_trans = []

    # Add fraud pattern : large amount
    fraudster = LargeAmountFraudster()
    fraud_count = round(genuine_transaction_count * (FRAUD_PATTERN_LA_CASE_PERCENTAGE / 100))
    fraud_trans = fraudster.generate_transactions(customer_ids, terminal_ids, fraud_count, FRAUD_PATTERN_LA_SUCCESSIVE_PROBABILITY, FRAUD_PATTERN_LA_AMOUNT_RANGE, TOTAL_SIMULATION_DAYS, random_state = LARGE_AMOUNT_FRAUD_GENERATION_RANDOM_SEED)
    more_fraud_trans += [ (*ft, 1, FRAUD_PATTERN_LA) for ft in fraud_trans ]

    # Add fraud pattern : card test
    fraudster = CardTestFraudster()
    fraud_count = round(genuine_transaction_count * (FRAUD_PATTERN_CT_CASE_PERCENTAGE / 100))
    fraud_trans = fraudster.generate_transactions(customer_ids, terminal_ids, fraud_count, FRAUD_PATTERN_CT_CARD_TEST_TERMINAL_COUNT, FRAUD_PATTERN_CT_SUCCESSIVE_PROBABILITY1, FRAUD_PATTERN_CT_SUCCESSIVE_PROBABILITY2, FRAUD_PATTERN_CT_AMOUNT_RANGE, TOTAL_SIMULATION_DAYS, random_state = CARD_TEST_FRAUD_GENERATION_RANDOM_SEED)
    more_fraud_trans += [ (*ft, 1, FRAUD_PATTERN_CT) for ft in fraud_trans ]

    # Add fraud pattern : successive small amount
    fraudster = SuccessiveSmallAmountFraudster()
    fraud_count = round(genuine_transaction_count * (FRAUD_PATTERN_SA_CASE_PERCENTAGE / 100))
    fraud_trans = fraudster.generate_transactions(customer_ids, terminal_ids, fraud_count, FRAUD_PATTERN_SA_SUCCESSIVE_PROBABILITY, FRAUD_PATTERN_SA_INTERVAL_RANGE, FRAUD_PATTERN_SA_AMOUNT_RANGE, TOTAL_SIMULATION_DAYS, random_state = SUCCESSIVE_SMALL_AMOUNT_FRAUD_GENERATION_RANDOM_SEED)
    more_fraud_trans += [ (*ft, 1, FRAUD_PATTERN_SA) for ft in fraud_trans ]

    # Add fraud pattern : merchant collusion
    fraudster = MerchantCollusionFraudster()
    fraud_count = round(genuine_transaction_count * (FRAUD_PATTERN_MC_CASE_PERCENTAGE / 100))
    fraud_trans = fraudster.generate_transactions(customer_profiles_table, terminal_ids, fraud_count, FRAUD_PATTERN_MC_COLLUDED_MERCHANT_COUNT, FRAUD_PATTERN_MC_SUCCESSIVE_PROBABILITY, FRAUD_PATTERN_MC_AMOUNT_RANGE, TOTAL_SIMULATION_DAYS, random_state = MERCHANT_COLLUSION_FRAUD_GENERATION_RANDOM_SEED)
    for tt in fraud_trans:
        more_fraud_trans += [ (*tt.pop(0), 0, 0) ]
        more_fraud_trans += [ (*ft, 1, FRAUD_PATTERN_MC) for ft in tt ]

    # Create a dataframe and add TX_DATETIME column
    more_fraud_trans_df = pd.DataFrame(more_fraud_trans, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'])
    if len(more_fraud_trans_df) > 0:
        more_fraud_trans_df['TX_DATETIME'] = pd.to_datetime(more_fraud_trans_df["TX_TIME_SECONDS"], unit='s', origin = SIMULATION_START_DATE)
        more_fraud_trans_df = more_fraud_trans_df[['TX_DATETIME','CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS', 'TX_FRAUD', 'TX_FRAUD_SCENARIO']]

    # Apppend fraud trans to transactions_df
    transactions_df = transactions_df.append(more_fraud_trans_df)
    
    # Sort by tx datetime
    transactions_df.sort_values('TX_DATETIME', inplace = True)

    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    
    # Output dataset 

    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    if os.path.exists(DIR_OUTPUT + FULLSET_FILENAME):
        os.remove(DIR_OUTPUT + FULLSET_FILENAME)

    if os.path.exists(DIR_OUTPUT + CUSTOMER_PROFILE_FILENAME):
        os.remove(DIR_OUTPUT + CUSTOMER_PROFILE_FILENAME)

    if os.path.exists(DIR_OUTPUT + TERMINAL_PROFILE_FILENAME):
        os.remove(DIR_OUTPUT + TERMINAL_PROFILE_FILENAME)

    customer_profiles_table.to_csv(DIR_OUTPUT + CUSTOMER_PROFILE_FILENAME, index = False)
    terminal_profiles_table.to_csv(DIR_OUTPUT + TERMINAL_PROFILE_FILENAME, index = False)
    transactions_df.to_csv(DIR_OUTPUT + FULLSET_FILENAME, index = False)


if __name__ == "__main__":
    main()

