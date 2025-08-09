# Example Python codebase for testing

def helper_function(x):
    """A simple helper function"""
    return x * 2

def another_helper(y):
    """Another helper function"""
    return y + 1

def main_function(a, b):
    """Main function that uses helper functions"""
    result1 = helper_function(a)
    result2 = another_helper(b)
    return result1 + result2

class Calculator:
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return helper_function(x) * y
    
    def complex_operation(self, a, b, c):
        """Complex operation using multiple dependencies"""
        temp1 = self.add(a, b)
        temp2 = self.multiply(temp1, c)
        temp3 = another_helper(temp2)
        return main_function(temp3, temp1)
