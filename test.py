class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        #self.orometer_reading = 0
    
    #@property 
    def orometer_reading(self):
        return 0
    
    def get_description(self):
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name
 
    def read_odometer(self):
        print("This car has "+ str(self.orometer_reading) + " miles on it")
 
 
my_new_car = Car("aodi", "a6", 2017)

print(my_new_car.orometer_reading())

 
 
#直接更改,修改初始化中的属性--
my_new_car.orometer_reading = 14
print(my_new_car.orometer_reading)

my_new_car.read_odometer()
 