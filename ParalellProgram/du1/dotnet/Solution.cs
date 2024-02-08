using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;

namespace dns_netcore
{
	class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient dnsClient;
        private ConcurrentDictionary<string, Domain> cacheResolvingDomains = new ConcurrentDictionary<string, Domain>();
		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
		}

        Mutex mutex = new Mutex();
		public async Task<IP4Addr> ResolveRecursive(string domain)
		{
			return await Task<IP4Addr>.Run(async () => {

                //mutex.WaitOne();
                //Console.WriteLine( );
                //Console.WriteLine( );
                //Console.WriteLine(domain);
                string[] domains = domain.Split('.');
				Array.Reverse(domains);

				(IP4Addr currentDomainIPAddress, string currentDomainName, int position) = await getBestIPAddressBase(domain, domains.Length);
                
                position = domains.Length - position;
                //Console.WriteLine( "LETS DO A TASK AT POSITION: " + position);

                if (position < 0 || position > domains.Length)
                {
                    throw new Exception("POSITION TOO SMALL OR TO BIG!");
                }
                
                for (int i = position; i < domains.Length; i++) // lets all chain of not cached values
                {
                    currentDomainIPAddress = getIpAdderess(currentDomainIPAddress, domains[i], currentDomainName);
                    if(i + 1 < domains.Length)
                        currentDomainName = domains[i + 1] + "." + currentDomainName;
                }

                //mutex.ReleaseMutex();
                return currentDomainIPAddress;
			});
		}

        /// <summary>
        /// Finds highest domain cached value if exists and awaits for the value. 
        /// </summary>
        /// <param name="domainName"></param>
        /// <param name="subDomainsCount"></param>
        /// <returns>
        /// values that are usefull for next task, the ip address of the server, 
        /// the domain and the position of the domain in the list of subdomains
        /// </returns>
        private async Task<(IP4Addr, string, int)> getBestIPAddressBase(string domainName, int subDomainsCount)
        {

            //Console.WriteLine("lets get best base --------");
            int position = 1;
            String lastDomainName = "";
            while(position <= subDomainsCount)
            {
                //Console.WriteLine("substring = " + domainName + "    position = " + position);

                if(cacheResolvingDomains.TryGetValue(domainName, out Domain domain))
                {
                    //Console.WriteLine("value CACHED: " + domain.name);
                    
                    var task = domain.getTask(statistics);

                    if(task.IsCompleted){
                        IP4Addr ip = await task;
                        
                        try
                        {
                            String checkDomain = await dnsClient.Reverse(ip);
                            if(checkDomain.Equals(domainName))
                                return (ip, lastDomainName, position - 1);
                            else
                            {
                                //Console.WriteLine("IPADDRESS DOESNT satisfy the server and required domain: " + checkDomain);
                            }
                        }
                        catch (DNSClientException)
                        {
                            //Console.WriteLine("SERVER NOT FOUND ++++++++");
                        }

                    } else {
                        //Console.WriteLine("value IN PROGRESS");
                        return (await task, lastDomainName, position - 1);
                    }
                    
                }

                // lets remove one prefixed domain
                lastDomainName = domainName;
                int dotIndex = domainName.IndexOf('.', StringComparison.Ordinal);
                if(dotIndex != -1){
                    domainName = domainName.Substring(dotIndex + 1, domainName.Length - 1 - dotIndex);
                }
                position++;
            }
            return (dnsClient.GetRootServers()[0], domainName, position - 1);
        }

        private IP4Addr getIpAdderess(IP4Addr serverIp, string domain, string name)
        {
            var tResult = dnsClient.Resolve(serverIp, domain);

            Domain newDomain = new Domain(name, tResult);
            newDomain = addToCacheResolvingDomains(newDomain.name, newDomain);

            tResult.Wait();
            return tResult.Result;
        }

        private static readonly int MAX_CACHE_SIZE = 1024;

        private SafeSortedDictionary<int, ConcurrentBag<Domain>> statistics = new SafeSortedDictionary<int, ConcurrentBag<Domain>>();
        private Domain addToCacheResolvingDomains(string name, Domain domain)
        {
            Domain addedDomain = cacheResolvingDomains.GetOrAdd(name, domain);
            
            if(addedDomain != domain)
            {
                //Console.WriteLine("CACHE REUSED: " + addedDomain.name);
            }
            else
            {
                //Console.WriteLine("CACHE UPDATED WITH: " + addedDomain.name);
            }
            if (addedDomain != domain && cacheResolvingDomains.Count > MAX_CACHE_SIZE){
                // remove the least used domain
                try
                {
                    if(statistics.First().TryTake(out Domain removeDomain)){
                        // success
                    }
                }
                catch (InvalidOperationException)
                {
                }
            }
            return addedDomain;
        }
    }

    class Domain
    {
        public readonly String name;
        private int _usageCount = 0;
        public int UsageCount 
        {
            get { return Volatile.Read(ref _usageCount); }
            set { Volatile.Write(ref _usageCount, value); }
        }

        private Task<IP4Addr> _task;

        public Task<IP4Addr> getTask(SafeSortedDictionary<int, ConcurrentBag<Domain>> statistics){
            statistics.PerformActionForKey(UsageCount, value => {
                Domain domain = this;
                value.TryTake(out domain);
            });
            UsageCount++;
            statistics.PerformActionForKey(UsageCount, value => value.Add(this));
            return _task;
        }


        public Domain(string name, Task<IP4Addr> task)
        {
            this.name = name;
            this._task = task;
        }
    }

    class SafeSortedDictionary<TKey, TValue> where TKey : IComparable<TKey>
    {
        private SortedDictionary<TKey, TValue> dictionary = new SortedDictionary<TKey, TValue>();
        private object syncRoot = new object();

        public void Add(TKey key, TValue value)
        {
            lock (syncRoot)
            {
                dictionary.Add(key, value);
            }
        }

        public bool TryGetValue(TKey key, out TValue value)
        {
            lock (syncRoot)
            {
                return dictionary.TryGetValue(key, out value);
            }
        }

        public void Remove(TKey key)
        {
            lock (syncRoot)
            {
                dictionary.Remove(key);
            }
        }
        public void Iterate(Action<KeyValuePair<TKey, TValue>> action)
        {
            lock (syncRoot)
            {
                foreach (KeyValuePair<TKey, TValue> pair in dictionary)
                {
                    action(pair);
                }
            }
        }

        public void Update(TKey key, TValue value)
        {
            lock (syncRoot)
            {
                if (dictionary.ContainsKey(key))
                {
                    dictionary[key] = value;
                }
                else
                {
                    dictionary.Add(key, value);
                }
            }
        }

        public void PerformActionForKey(TKey key, Action<TValue> action)
        {
            lock (syncRoot)
            {
                if (dictionary.TryGetValue(key, out TValue value))
                {
                    action(value);
                }
            }
        }

        public TValue First()
        {
            lock (syncRoot)
            {
                if (dictionary.Count == 0)
                {
                    throw new InvalidOperationException("Dictionary is empty.");
                }
                
                return dictionary.First().Value;
            }
        }
    }
}