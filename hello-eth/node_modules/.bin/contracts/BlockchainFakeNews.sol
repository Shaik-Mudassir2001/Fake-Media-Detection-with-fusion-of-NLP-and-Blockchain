pragma solidity >= 0.8.11 <= 0.8.11;

contract BlockchainFakeNews {
    string public signup_details;
    string public publish_news;
    
       
    //call this function to save new user details data to Blockchain
    function setSignup(string memory sd) public {
       signup_details = sd;	
    }
   //get Signup details
    function getSignup() public view returns (string memory) {
        return signup_details;
    }

    //call this function to save publish news to Blockchain
    function setPublishNews(string memory pt) public {
       publish_news = pt;	
    }
   //get publish news details
    function getPublishNews() public view returns (string memory) {
        return publish_news;
    }

   constructor() public {
        signup_details="";
	publish_news="";
    }
}