//
//  common.h
//  Opt
//
//  Created by Sashan Jakkam Reddi on 2/18/16.
//  Copyright (c) 2016 Sashan Jakkam Reddi. All rights reserved.
//

#ifndef Opt_common_h
#define Opt_common_h

#include <cassert>
#include <iostream>
#include <string>

//extern std::string g_log_tag;

// Sets the tag string that is output with each log statement.
// This useful for e.g. printing experiment name to keep track of which experiment
// is currently running.
//#define SET_LOG_TAG(x) g_log_tag = (x)

#define LOG(x) std::cerr << "LOG[" << "|" << __FILE__ << ":" << __LINE__ << "]: " << x << std::endl

#define ASSERT(condition, msg)  {if(!(condition)) {std::cerr << "ASSEERTION FAILED: " << #condition << std::endl << "MESSAGE: " << msg << std::endl; assert(false);}}

#define ASSERT_WITHIN(x, lower, upper, msg) ASSERT((x) >= (lower) && (x) <= (upper), msg)
#define ASSERT_NEAR(x, target, tolerance, msg) ASSERT_WITHIN((x), (target)-(tolerance), (target)+(tolerance), msg)


#endif
