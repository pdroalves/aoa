#include <AOADGT/tool/version.h>

std::string GET_AOADGT_VERSION() {     
    std::ostringstream oss; 
    oss << AOADGT_VERSION_MAJOR << "." << AOADGT_VERSION_MINOR << "." << AOADGT_VERSION_PATCH; 
    if(AOADGT_VERSION_TWEAK != 0)
	    oss << " - " << AOADGT_VERSION_TWEAK;
    return oss.str();
}
