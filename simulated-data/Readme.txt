Customer count 5000
Terminal count 2500
Simulation days 365
customer terminal radius 30

NOTE
----
*NOT ALL* columns in fe_trans.csv are needed for training.
e.g. TRANSACTION_ID, CUSTOMER_ID, TERMINAL_ID are definitely not useful. TX_DATETIME cannot be fed into any model.
Please refer to below for columns for training. You can drop more columns if needed.


#   Column                                            scaling
---  ------                                            -----         
 4   TX_AMOUNT                                         StandardScaler
 5   TX_TIME_SECONDS                                   StandardScaler
 6   TX_TIME_DAYS                                      StandardScaler
 7   TX_FRAUD                                          this is the target variable
               
 10  TX_LAST_SECONDS                                   StandardScaler
 11  TX_LAST_HOURS                                     StandardScaler
 12  TX_LAST_DAYS                                      StandardScaler
 13  TX_TIME_HOUR_BIN_0                                
 14  TX_TIME_HOUR_BIN_1                                
 15  TX_TIME_HOUR_BIN_2                                
 16  TX_TIME_HOUR_BIN_3                                
 17  TX_TIME_HOUR_BIN_4                                
 18  TX_TIME_HOUR_BIN_5                                
 19  CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW                StandardScaler
 20  CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW                StandardScaler
 21  CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW               StandardScaler
 22  CUSTOMER_ID_AVG_AMOUNT_2REC                       StandardScaler
 23  CUSTOMER_ID_AVG_AMOUNT_10REC                      StandardScaler
 24  TERMINAL_ID_RISK_2DAY_WINDOW                      StandardScaler
 25  TERMINAL_ID_RISK_7DAY_WINDOW                      StandardScaler
 26  SUM_TX_AMOUNT_CUSOMTER_ID_SAME_TERMINAL_SAME_DAY  StandardScaler
 27  nb_TX_CUSOMTER_ID_SAME_TERMINAL_SAME_DAY          StandardScaler
 28  ONE_DOLLAR                                        
 29  CUSTOMER_TERMINAL_DISTANCE                        StandardScaler
 30  AMOUNT_Z_SCORE                                    StandardScaler
 31  CUSTOMER_TERMINAL_DISTANCE_Z_SCORE                StandardScaler