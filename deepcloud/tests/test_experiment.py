import boto3, subprocess
import datetime
import time
import pandas as pd
import numpy as np
import fabric
from fabric.api import run, hide

class benchmark:
    def __init__(self, experiments, instance_type, image_id="ami-a3b3d4b5", key_name='dpcld_test1'):
        self.experiments = experiments
        self.instance_type = instance_type
        self.image_id = image_id
        self.key_name = key_name
        self.run_date = datetime.date.strftime(datetime.date.today(), '%Y%m%d')
        self.client = boto3.client('ec2')
        self.ec2 = boto3.resource('ec2')
        self.p_ids = []
        #self.instances = self.setUpInstances()

        #self.configureInstances()

    def setUpInstances(self):
        num_inst = len(experiments)

        response = self.client.run_instances(ImageId=self.image_id,
                                InstanceType=self.instance_type,
                                MinCount=num_inst,
                                MaxCount=num_inst,
                                KeyName=self.key_name)

        assert response.get('ResponseMetadata').get('HTTPStatusCode') == 200, "Request ended with an error (HTTPStatusCode != 200)"
        assert len(response.get('Instances')) == num_inst, "Number of instances launched is equal to specified"

        instance_ids = []
        instance_ids.extend([_.get('InstanceId') for _ in response.get('Instances')])

        waiter = self.client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=instance_ids)

        print('Set up instances: {}'.format(' '.join(instance_ids)))

        self.instances = [self.ec2.Instance(_) for _ in instance_ids]

        #return [self.ec2.Instance(_) for _ in instance_ids]

    def configureInstances(self):
        [self.configureInstance(i, e) for i, e in zip(self.instances, self.experiments)]

    def configureInstance(self, instance, experiment):
        instance.create_tags(Tags=[{'Key':'Name', 'Value':'{}_{} experiment'.format(*experiment)}])
        fabric.api.env.host_string = 'ec2-user@{}'.format(instance.public_dns_name)
        fabric.api.env.key_filename = '~/.ssh/{}.pem'.format(self.key_name)
        fabric.api.env.disable_known_hosts
        fabric.api.hide('output')
        fabric.operations.put('~/Studia/mgr/deepcloud/tests/test_{}_{}.py'.format(*experiment), '.')
        run('sudo pip3 install git+git://github.com/fchollet/keras.git --upgrade')
        run('sudo pip3 install pydot')
        run('sudo pip3 install graphviz')

    def runExperiments(self):
        [self.runExperiment(i, e) for i, e in zip(self.instances, self.experiments)]

    def runExperiment(self, instance, experiment):
        fabric.api.env.host_string = 'ec2-user@{}'.format(instance.public_dns_name)
        run('nohup python3 -u test_{0}_{1}.py \
        --dataset {0} --architecture {1} --run_date {2} \
        >{2}_{0}_{1}.log 2>{2}_{0}_{1}.err < /dev/null &'.format(*experiment, self.run_date), pty=False)
        self.p_ids.append((instance.id, run('pgrep -f "python3 -u test"')))

    def getExperimentLogs(self):
        running_instances = self.instances

        while running_instances:
            for instance in self.instances:
                fabric.api.env.host_string = 'ec2-user@{}'.format(instance.public_dns_name)

                status = benchmark_td1.client.describe_instance_status(InstanceIds=[instance.id])\
                .get('InstanceStatuses')[0]\
                .get('InstanceState')\
                .get('Name')

                if status != 'running':
                    print('Instance {} not running'.format(instance.id))
                    running_instances.remove(instance)
                    continue

                rSync(instance.public_dns_name)

                try:
                    run('pgrep -f "python3 -u test"')
                except:
                    print('Process on instance {} not found'.format(instance.id))
                    running_instances.remove(instance)
                    continue

                time.sleep(1)

            print('{}: synchronized logs from all instances'.format(datetime.datetime.now().isoformat()))
        else:
            print('All instances are down or finished their tasks')

    def rSync(self, public_dns):
        fabric.operations.local('rsync -aL -e "ssh -i ~/.ssh/{}.pem -o StrictHostKeyChecking=no" \
        --include="/home/ec2-user/*" --include="*.log" --include="*.err" \
        --include="*.out" --include="*.png" --exclude="*" \
        ec2-user@{}:/home/ec2-user/ \
        ~/Studia/mgr/deepcloud/tests/logs/'\
                                .format(self.key_name, public_dns))

experiments = [("mnist","kerasdef")]

benchmark_td1 = benchmark(experiments=experiments,instance_type='t2.micro')
benchmark_td1.setUpInstances()
benchmark_td1.configureInstances()
benchmark_td1.runExperiments()
benchmark_td1.getExperimentLogs()

term_responses = [_.terminate() for _ in benchmark_td1.instances]
