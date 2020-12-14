import pyodbc
from datetime import datetime

# CONNECTIONSTRING LOKAL DATABASE

# Definition of connection string used to connect to the database
connString = 'Trusted_Connection=yes;' \
             'Driver={ODBC Driver 17 for SQL Server};' \
             'SERVER=localhost;' \
             'PORT=1433;' \
             'DATABASE=CommentorDB;' \
             'UID=CommentorAdmin;' \
             'PWD=Admin123'

# Gets the latest Car amount from the database Cars table
def getLatestVehicleAmount(vehicleType):
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    now = datetime.now()
    hourStr = now.strftime("%H")
    sql = "SELECT SUM(VehicleAmount) AS VehicleAmount from Vehicle WHERE TypeName = '%s' and FORMAT(DateTimeStamp,'HH') = '%s'" % (vehicleType, hourStr)
    cursor.execute(sql)
    for row in cursor:
        if row[0] is not None:
            return row[0]
        else:
            return 0


# Updates the latest row in Cars table
def updateRow(vehicleType):
    try:
        cn = pyodbc.connect(connString, autocommit=True)
        latestRow = getLatestRow()
        newNumber = getLatestVehicleAmount(vehicleType) + 1
        now = datetime.datetime.now()
        dateStr = now.strftime("%d %b %Y")
        hourStr = now.strftime("%H")
        sql = "Update Vehicle set VehicleAmount = %s WHERE TypeName = '%s' and DateTimeStamp = %s and" % (newNumber, vehicleType, hourStr, dateStr)

        cursor = cn.execute(sql)
        cn.commit()
    except Exception as e:
        print('Error connecting to database: ', str(e))

    finally:
        cn.close()
        print("Updated VehicleId ("+str(vehicleType)+"): " + str(latestRow) + " to Amount: " + str(getLatestVehicleAmount(vehicleType)))

# Inserts a new row into the Cars table
def insertRow(number, check, vehicleType):
    try:
        cn = pyodbc.connect(connString, autocommit=True)
        now = datetime.now()
        formatted_date = now.strftime('%d %b %Y %H:%M:%S')
        feed = "1"
        sql2 = "INSERT INTO Vehicle(TypeName, VehicleAmount, Feed, DateTimeStamp) VALUES ('%s', %d, %s, '%s') " % (vehicleType, number, feed, formatted_date)
        cursor2 = cn.execute(sql2)
        cn.commit()

    except Exception as e:
        print('Error connecting to database: ', str(e))

    finally:
        cn.close()
        if check:
            print("Table empty, inserted new row")
        else:
            print("Inserted New Row")
