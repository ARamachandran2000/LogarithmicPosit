from copy import *
from math import *
from fractions import Fraction
from decimal import Decimal, getcontext
from ctypes import c_ulonglong, c_double
from BitUtils import *
# from FixedPoint import *
from functools import reduce
import operator
import torch
import numpy as np

class LogarithmicPosit(object):
    def __init__(self, number = 0, nbits = None, es = None, rs = None, sf = None) -> None:
        if (nbits == None or es == None):
            raise Exception("Nbits and es are required")
        
        # If no values are provided for rs adn sf, assume standard posit values
        elif(rs == None and (sf == None or sf == 0)):
            self.nbits = nbits
            self.es = es
            self.rs = nbits - 1
            self.sf = 0

        else:
            self.nbits = nbits
            self.es = es
            self.rs = rs
            self.sf = sf

        self.number = 0

        # Calculate number of patterns
        self.npat = 2**self.nbits

        # Standard Posit Useed
        self.useed = 2**2**self.es

        self.ulfx = 0

        # Check min and max ranges
        if(self.rs < (self.nbits - 1)):
            intm = 2**(self.nbits - 1) - 1
            self.maxpos = intm & unsetBit(intm, (self.nbits - 1 - self.rs))
            self.minpos = 1 << (self.nbits - 1 - self.rs)

        else:
            self.maxpos = 2**(self.nbits - 1) - 1
            self.minpos = 1
        
        # Calculate Standard Posit Infinity and Zero
        self.inf = 2**(self.nbits - 1)
        self.zero = 0

        if type(number) == str:
            self.set_string(number)
        elif type(number) == int:
            self.set_int(number)
        else:
            self.set_float(float(number))
        
        self.log_pos_number = self.encode_log_pos()
    
    def float_to_int(self, n):
        if type(n) == float:
            return c_ulonglong.from_buffer(c_double(n)).value
        else:
            raise "Not float"   

    def set_bit_pattern(self, x):
        if type(x) == str:
            if x.count("1") + x.count("0") == len(x):
                if len(x) <= self.nbits:
                    self.number = int(x, 2)
                else:
                    raise "String length exceeds number of bits"
            else:
                raise "String must contain only 1 and 0's"
        
        elif type(x) == int:
            if countBits(x) <= self.nbits:
                self.number = x
            else:
                raise "Integer exceeds number of bits"
        else:
            raise "Not string or int"   


    def set_float(self, x):
        if type(x) == float:
            # (1) negative or positive zero -> return zero posit 
            if x == 0:
                self.number = self.zero
            # (2) +-inf or NaN -> return posit infinity
            elif isinf(x) or isnan(x):
                self.number = self.inf
            # (3) normal float
            else:
                # convert to integer
                n = self.float_to_int(x)
                # to get sign bit, shift 63 times to the right
                sign = n >> 63
                # to get exponent bits, remove sign, shift, then subtract bias
                exponent = ((n & ((1 << 63) - 1)) >> 52) - 1023  
                # to get fractions bits, mask fraction bits and then OR the hidden bit
                fraction = (1 << 52) | (n & ((1 << 52) - 1))
                # given the decoded values, construct a posit
                self.number = self.construct(sign, exponent, fraction).number
        else:
            raise "Not Float"

    def set_int(self, x):
        if type(x) == int:
            if x == 0:
                self.number = 0
            else:
                sign = 0 if x >= 0 else 1
                if sign == 1:
                    x = abs(x)
                exponent = countBits(x) - 1
                fraction = x
                self.number = self.construct(sign, exponent, fraction).number
        else:
            raise "Not an integer"

    def set_string(self, x):
        if type(x) == str:
            if len(x) == 0:
                return "Empty string"
            dot_index = x.find('.')
            sign = int(x[0] == "-")
            if dot_index == -1:
                self.set_int(int(x))
            elif dot_index == len(x) - 1:
                self.set_int(int(x[:-1]))
            else:
                if sign == 1:
                    x = x[1:]
                    dot_index -= 1
                # count number of fractional digits
                fdig = len(x) - 1 - dot_index
                # get fraction
                fraction = int(x[:dot_index] + x[dot_index+1:])
                exponent = countBits(fraction) - 1 - fdig
                self.number = (self.construct(sign, exponent, fraction) / LogarithmicPosit(5**fdig, nbits = self.nbits, es = self.es)).number
        else:
            return "Not string"

    def get_value(self):
        # 50 digits of precision
        getcontext().prec = 100
        if self.number == 0:
            return Decimal("0")
        elif self.number == self.inf:
            return Decimal("inf")

        sign, regime, exponent, fraction = self.decode_log_pos() 

        f = Decimal(fraction)
        n = countBits(int(fraction)) - 1

        return ((-1)**sign * Decimal(2)**Decimal(2**self.es * regime + exponent - n) * Decimal(f))

    def __float__(self):
        return float(self.get_value())

    def construct(self, sign, scale, fraction):
        if fraction == 0:
            return LogarithmicPosit(nbits = self.nbits, es = self.es, rs = self.rs, sf=self.sf)
    
        n = 0

        # If using generalised Posits
        if(self.rs < (self.nbits - 1)):
            max_regime_length = self.rs
            max_regime_value = (self.rs - 1) - 1
            min_regime_value = -(self.rs - 1)

            regime = scale >> self.es

            if(regime > max_regime_value):
                regime = max_regime_value
                regime_length = max_regime_length
                if(max_regime_value == 0):
                    exponent = scale
                else:
                    exponent = scale % max_regime_value
            
            elif(regime < min_regime_value):
                regime = min_regime_value
                regime_length = max_regime_length
                # exponent is always unsigned
                exponent = abs(scale) % abs(regime)
            
            else:
                regime = regime
                regime_length = regime + 2 if regime >= 0 else -regime + 1
                exponent = scale & createMask(self.es, 0)
            
            if(exponent > (2**self.es - 1)):
                print("Overflow regime =", regime, bin(self.maxpos))
                p = LogarithmicPosit(nbits = self.nbits, es = self.es, rs = self.rs, sf = self.sf)
                p.set_bit_pattern(self.maxpos if regime >= 0 else self.minpos)
                if sign == 1:
                    p = -p
                return p

        else:
            regime = scale >> self.es
            exponent = scale & createMask(self.es, 0)
            # number of bits written for regime
            regime_length = regime + 2 if regime >= 0 else - regime + 1

            # overflow to maxpos underflow to minpos
            if regime_length >= self.nbits:
                p = LogarithmicPosit(nbits = self.nbits, es = self.es, rs = self.rs, sf = self.sf)
                p.set_bit_pattern(self.maxpos if regime >= 0 else self.minpos)
                if sign == 1:
                    p = -p
                return p

        # encode regime
        if regime >= 0:
            n |= createMask(regime_length - 1, self.nbits - regime_length)
        elif self.nbits - 1 >= regime_length:
            n |= setBit(n, self.nbits - 1 - regime_length)
        # count number of bits available for exponent and fraction
        exponent_bits = min(self.es, self.nbits - 1 - regime_length)
        fraction_bits = self.nbits - 1 - regime_length - exponent_bits
    
        # remove trailing zeroes
        fraction = removeTrailingZeroes(fraction)
        # length of fraction bits, -1 is for hidden bit
        fraction_length = countBits(fraction) - 1
        # remove hidden bit
        fraction &= 2**(countBits(fraction)-1) - 1
        # trailing_bits = number of bits available for exponent + fraction
        trailing_bits = self.nbits - 1 - regime_length
        # exp_frac = concatenate exponent + fraction without trailing zeroes
        exp_frac = removeTrailingZeroes(exponent << (fraction_length) | fraction)
        # exp_frac_bits = minimum number of bits needed to represent exp_frac
        # exponent only
        if fraction_length == 0:
            exp_frac_bits = self.es - countTrailingZeroes(exponent)
        # exponent plus fraction
        else:
            exp_frac_bits = self.es + fraction_length
        
        # rounding needs to be done
        if trailing_bits < exp_frac_bits:
            # get overflow bits
            overflown = exp_frac & createMask(exp_frac_bits - trailing_bits, 0)
            # truncate trailing bits, encode to number
            n |= exp_frac >> (exp_frac_bits - trailing_bits)
            # perform round to even rounding by adding last bit to overflown bit
            # tie-breaking
            if overflown == (1 << (exp_frac_bits - trailing_bits - 1)):
                # check last bit
                if checkBit(exp_frac, exp_frac_bits - trailing_bits):
                    n += 1
            # round to next higher value
            elif overflown > (1 << (exp_frac_bits - trailing_bits - 1)):
                n += 1
            # round to next lower value
            else:
                None
        else:
            n |= exp_frac << (trailing_bits - exp_frac_bits)
            
        p = LogarithmicPosit(nbits = self.nbits, es = self.es, rs = self.rs, sf = self.sf)
        if sign == 0:
            p.set_bit_pattern(n)
        else:
            p.set_bit_pattern(twosComplement(n, self.nbits))

        return p


    def decode(self):
        x = self.number

        # exception values
        if x == 0:
            return (0, 0, 0, 0)
        elif x == self.inf:
            return None
        
        # determine sign and decode
        sign = checkBit(x, self.nbits - 1)
        
        if sign == 1:
            x = twosComplement(x, self.nbits)

        # decode regime length and regime sign
        regime_sign = checkBit(x, self.nbits - 2) 
        if regime_sign == 0:
            regime_length = self.nbits - lastSetBit(x) - 1
        else:
            regime_length = self.nbits - lastUnsetBit(x) - 1
        
        # determine lengths
        exponent_length = max(0, min(self.es, self.nbits - 1 - regime_length))
        fraction_length = max(0, self.nbits - 1 - regime_length - exponent_length)
        
        # determine actual values
        regime = - regime_length + 1 if regime_sign == 0 else regime_length - 2
        exponent = extractBits(x, exponent_length, fraction_length) << (self.es - exponent_length)
        
        fraction = removeTrailingZeroes(setBit(extractBits(x, fraction_length, 0), fraction_length))

        return (sign, regime, exponent, fraction)

    def encode_log_pos(self):
        sign, regime, exponent, fraction = self.decode()
        self.ulfx = exponent + (0 if fraction == 0 else log2(fraction))
        scale = 2**(regime - self.sf)
        return (1 if sign == 0 else -1) * scale * 2**(self.ulfx)

    def decode_log_pos(self):
        x = self.log_pos_number
        _,regime,exponent, _ = self.decode()
        
        # determine sign and decode
        sign = (0 if x >= 0 else 1)
        # determine actual values
        fraction = exp(self.ulfx - exponent)
        return (sign, regime, exponent, fraction)