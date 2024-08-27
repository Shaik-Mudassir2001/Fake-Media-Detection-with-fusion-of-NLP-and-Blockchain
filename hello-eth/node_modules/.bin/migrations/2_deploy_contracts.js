const BlockchainFakeNews = artifacts.require("BlockchainFakeNews");

module.exports = function(deployer) {
  deployer.deploy(BlockchainFakeNews);
};
