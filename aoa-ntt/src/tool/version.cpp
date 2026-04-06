#include <newckks/tool/version.h>

std::string GET_VERSION() {     
    std::ostringstream oss; 
    oss << NEWCKKS_VERSION_MAJOR << "." << NEWCKKS_VERSION_MINOR << "." << NEWCKKS_VERSION_PATCH; 
    if(NEWCKKS_VERSION_TWEAK != 0)
	    oss << " - " << NEWCKKS_VERSION_TWEAK;
    return oss.str();
}
